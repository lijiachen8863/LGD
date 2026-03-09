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

import json
import numpy as np
from PIL import Image

_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]

def convert_PIL_to_numpy(image, format):
        """
        Convert PIL image to numpy array of target format.

        Args:
            image (PIL.Image): a PIL image
            format (str): the format of output image

        Returns:
            (np.ndarray): also see `read_image`
        """
        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format in ["BGR", "YUV-BT.601"]:
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)

        # handle formats not supported by PIL
        elif format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        elif format == "YUV-BT.601":
            image = image / 255.0
            image = np.dot(image, np.array(_M_RGB2YUV).T)

        return image

def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])



def get_mask_from_panoptic(seg_img_path, seg_id):
    seg_img=Image.open(seg_img_path)

    seg_map = convert_PIL_to_numpy(seg_img, "RGB")
    seg_map= rgb2id(seg_map)
    mask = seg_map == seg_id
    
    return mask


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


class DuMoGaTestDataset(Dataset):
    def __init__(self, data_dir, split, input_size,
                 word_length, coco_path):
        super(DuMoGaTestDataset, self).__init__()
        self.mode = "test"
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        
        annotations_path = os.path.join(data_dir, "sub_ris.json")
        with open(annotations_path, "r") as r:
            annotations= json.loads(r.read())

        self.annotations = annotations
        ref_ids = range(len(annotations))
        # get all image in the train/val/test split
        self.coco_path = coco_path
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

        this_annotation = self.annotations[this_ref_id]
        
        this_img = os.path.join(self.coco_path, this_annotation["file_name"])

        img = Image.open(this_img).convert("RGB")

        this_comments = [this_annotation["sentence"]]

        this_mask = get_mask_from_panoptic(os.path.join(self.coco_path, this_annotation["pan_seg_file"]), this_annotation["id"])
        # this_mask, this_comments, _ = get_mask_from_json(os.path.join(self.data_dir, self.split, f"{this_ref_id}.json"), img)
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
