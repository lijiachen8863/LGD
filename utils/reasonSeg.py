import glob
import json
import os
from enum import Enum
import torch.distributed as dist

import cv2
import numpy as np

import os
import sys
from torch.utils.data import Dataset
import torch
# import torch.nn.functional as F
from torchvision.transforms import functional as F

from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import utils.transforms as T
import random
import clip
import argparse
# import h5py


def get_transform(img_size, mode):
    transforms = [T.Resize(img_size, img_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]

    return T.Compose(transforms)


def get_mask_from_json(json_path, img):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    width, height= img.size

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class resonSegTestDataset(Dataset):
    def __init__(self, data_dir, split, input_size,
                 word_length):
        super(resonSegTestDataset, self).__init__()
        self.mode = "test"
        self.input_size = (input_size, input_size)
        self.word_length = word_length

        ref_ids = []
        for json_file in os.listdir(os.path.join(data_dir, split)):
            if json_file.endswith(".json"):
                ref_ids.append(json_file.split(".json")[0])

        # get all image in the train/val/test split
        self.data_dir = data_dir
        self.split = split
        self.ref_ids = ref_ids
        #self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.transform = get_transform(input_size, self.mode)

    def __len__(self):
        # Different form other supervised works, 
        # length of the dataset equals to the number 
        # of images under weakly supervised setting
        return len(self.ref_ids)

    def __getitem__(self, index):
        # To reproduce the method in the weakly supervised setting, the dataloader picks 
        # images instead of referring expressions during training.
        #pdb.set_trace()

        this_ref_id = self.ref_ids[index]
        this_img = os.path.join(self.data_dir, self.split, f"{this_ref_id}.jpg")
        img = Image.open(this_img).convert("RGB")

        this_mask, this_comments, _ = get_mask_from_json(os.path.join(self.data_dir, self.split, f"{this_ref_id}.json"), img)
        ori_img = img.copy()
        ori_img = F.to_tensor(ori_img)

        sents = this_comments
        sents_id = []
        for sent_index in range(len(sents)):
            sents_id.append(f"{this_ref_id}_{sent_index}")

        ref_mask = this_mask
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        img_size = ref_mask.shape[:2]
        

        if self.transform is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img, mask = self.transform(img,annot)

        params = {
            #'word_vecs': word_vecs,
            #'attention_masks': attention_masks,
            #'subj_index': selected_subj_index,
            #'attr_index': selected_attr_index,
            'masks': ref_mask, # use the mask in orginal size
            'ori_img': ori_img,
            'sents_id': sents_id,
            'ori_size': np.array(img_size),
            'sents': sents,
            'ref_id':this_ref_id,
            'file_name':this_img
        }
        return img, params
        

    def __repr__(self):
        return self.__class__.__name__ + "(" + \
            f"mode={self.mode}, " + \
            f"input_size={self.input_size}, " + \
            f"word_length={self.word_length}"
            #f"db_path={self.lmdb_dir}, " + \
