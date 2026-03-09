import argparse

import numpy as np
import json
import torch
from PIL import Image

# segment anything
import cv2

from tqdm import tqdm
#import wandb
#from utils.dataset import tokenize
# from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather,
#                         trainMetricGPU)
import argparse
import warnings
import torch.nn.parallel
import torch.utils.data

import spacy
#import clip

import torch
import cv2
import argparse
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
warnings.filterwarnings("ignore")
cv2.setNumThreads(0)
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
# from simple_tokenizer import SimpleTokenizer as _Tokenizer
# _tokenizer = _Tokenizer()
# from lavis.models import load_model_and_preprocess


import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

from utils.reasonSeg import resonSegTestDataset
from utils.DuMoGa import DuMoGaTestDataset

# dataset_name = "reasonSeg"
dataset_name = "DuMoGa"

def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    if dataset_name == "DuMoGa":
        parser.add_argument('--data-dir',
                            default='/home/ljc/datasets/DuMoGa/',
                            # default='/home/ljc/datasets/reasonSeg/',
                            type=str,
                            help='config file')
    elif dataset_name == "reasonSeg":
        parser.add_argument('--data-dir',
                            default='/home/ljc/datasets/reasonSeg/',
                            type=str,
                            help='config file')
    else:
        raise Exception("数据集未实现")
    parser.add_argument('--Weight_S_l',
                default=0.5,
                type=float,
                help='config file')
    parser.add_argument('--Weight_S_n',
                default=0.75,
                type=float,
                help='config file')
    parser.add_argument('--test-split',
                default="val",
                type=str,
                help='config file')
    parser.add_argument('--input-size',
                default=1024,
                type=int,
                help='config file')
    parser.add_argument('--word-len',
                default=50,
                type=int,
                help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    parser.add_argument
    args = parser.parse_args()
    # assert args.config is not None
    # cfg = config.load_cfg_from_cfg_file(args.config)
    # if args.opts is not None:
    #     cfg = config.merge_cfg_from_list(cfg, args.opts)
    # return cfg, args
    return args

def crop_image(image, mask):
    x, y, w, h = mask["bbox"]
    masked = image * np.expand_dims(mask["segmentation"], -1)
    crop = masked[y : y + h, x : x + w]
    if h > w:
        top, bottom, left, right = 0, 0, (h - w) // 2, (h - w) // 2
    else:
        top, bottom, left, right = (w - h) // 2, (w - h) // 2, 0, 0
    # padding
    crop = cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),)
    
    #padding_len = int(crop.shape[0]/8)
    #crop = cv2.copyMakeBorder(crop,padding_len,padding_len,padding_len,padding_len,cv2.BORDER_CONSTANT,value=(0,0,0))
    
    crop = Image.fromarray(crop)
    return crop

def crop_image_with_background_blur(image, mask):
    """
    We mask the image while blurring the background, currently we don't change the image size.
    """
    x, y, w, h = mask["bbox"]
    masked_img = cv2.bitwise_and(image, image, mask=np.uint8(mask["segmentation"]))
    if h > w:
        top, bottom, left, right = 0, 0, (h - w) // 2, (h - w) // 2
    else:
        top, bottom, left, right = (w - h) // 2, (w - h) // 2, 0, 0
    
    # Invert mask and blur background
    # blurred_img = cv2.GaussianBlur(image, (75, 75), 0, 0)
    blurred_img = cv2.GaussianBlur(image.copy(), [0, 0], sigmaX=50, sigmaY=50)
    blurred_background = cv2.bitwise_and(blurred_img, blurred_img, mask=np.uint8(1-mask["segmentation"]))

    # Combine masked image and blurred background
    result = cv2.add(masked_img, blurred_background)


    crop = result[y : y + h, x : x + w]
    #crop = blurred_background[y : y + h, x : x + w]
    crop = cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    crop = Image.fromarray(crop)
    return crop

def mask_blur_no_crop(image, mask):
    """
    add foreground as one of outputs
    We mask the image while blurring the background, currently we don't change the image size.
    """
    x, y, w, h = mask["bbox"]
    masked_img = cv2.bitwise_and(image, image, mask=np.uint8(mask["segmentation"]))
    if h > w:
        top, bottom, left, right = 0, 0, (h - w) // 2, (h - w) // 2
    else:
        top, bottom, left, right = (w - h) // 2, (w - h) // 2, 0, 0
    
    # Invert mask and blur background
    # blurred_img = cv2.GaussianBlur(image, (75, 75), 0, 0)
    blurred_img = cv2.GaussianBlur(image.copy(), [0, 0], sigmaX=50, sigmaY=50)
    blurred_background = cv2.bitwise_and(blurred_img, blurred_img, mask=np.uint8(1-mask["segmentation"]))

    # Combine masked image and blurred background
    result = cv2.add(masked_img, blurred_background)

    result = Image.fromarray(result)
    # return crop, Image.fromarray(masked_img)
    return result


def extract_target_np(sentence,nlp):
    # Parse the sentence with SpaCy
    doc = nlp(sentence)
    
    # Find the root word of the sentence
    root = next((token for token in doc if token.head == token), None)
    if root is None:
        return sentence  # use the whole sentence if no root word is found
    
    # If the root word is a verb, use its children noun as the root word
    if root.pos_ == 'VERB':
        for child in root.children:
            if child.pos_ == 'NOUN':
                root = child
                break
    
    # Find the noun phrase containing the root word
    target_np = next((np for np in doc.noun_chunks if root in np), None)
    if target_np is None:
        # If no noun phrase containing the root word is found, try to use the children of the root word instead
        children_nouns = [child for child in root.children if child.pos_ == 'NOUN']
        if len(children_nouns) == 1:
            target_np = next((np for np in doc.noun_chunks if children_nouns[0] in np), None)
    
    if target_np is None:
        return sentence  # use the whole sentence if still no target noun phrase is found
    
    return target_np.text

def extract_noun_phrase(text, nlp, need_index=False):
    # text = text.lower()

    doc = nlp(text)

    chunks = {}
    chunks_index = {}
    for chunk in doc.noun_chunks:
        for i in range(chunk.start, chunk.end):
            chunks[i] = chunk
            chunks_index[i] = (chunk.start, chunk.end)

    for token in doc:
        if token.head.i == token.i:
            head = token.head

    if head.i not in chunks:
        children = list(head.children)
        if children and children[0].i in chunks:
            head = children[0]
        else:
            if need_index:
                return text, [], text
            else:
                return text

    head_noun = head.text
    head_index = chunks_index[head.i]
    head_index = [i for i in range(head_index[0], head_index[1])]

    sentence_index = [i for i in range(len(doc))]
    not_phrase_index = []
    for i in sentence_index:
        not_phrase_index.append(i) if i not in head_index else None

    head = chunks[head.i]
    if need_index:
        return head.text, not_phrase_index, head_noun
    else:
        return head.text


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    # preprocess function used in clip
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

# def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
#     """
#     Returns the tokenized representation of given input string(s)
#     Parameters
#     ----------
#     texts : Union[str, List[str]]
#         An input string or a list of input strings to tokenize
#     context_length : int
#         The context length to use; all CLIP models use 77 as the context length
#     truncate: bool
#         Whether to truncate the text in case its encoding is longer than the context length
#     Returns
#     -------
#     A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
#     We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
#     """
#     if isinstance(texts, str):
#         texts = [texts]

#     sot_token = _tokenizer.encoder["<|startoftext|>"]
#     eot_token = _tokenizer.encoder["<|endoftext|>"]
#     all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
#     if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
#         result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
#     else:
#         result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

#     for i, tokens in enumerate(all_tokens):
#         if len(tokens) > context_length:
#             if truncate:
#                 tokens = tokens[:context_length]
#                 tokens[-1] = eot_token
#             else:
#                 raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
#         result[i, :len(tokens)] = torch.tensor(tokens)

#     return result

def calculate_overlap(masks, min_area=2000,min_iou=0.4):
    """
    There are redundant masks generated by sam, we filter them by keeping the larger masks
    masks: dict type, raw output of sam,
    min_area: int type, keep masks whose area larger than this,
    min_iou: dict type, if the overlap of a pair of masks greater than this, we keep the larger mask
    """
    overlap_areas = np.zeros((len(masks), len(masks)))
    area_large = []
    small_indices = []
    for i in range(len(masks)):
        area_large.append(np.sum(masks[i]["segmentation"]))
        if np.sum(masks[i]["segmentation"]) < min_area:
            small_indices.append(i)

    
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            if i not in small_indices and j not in small_indices:
                intersection = np.logical_and(masks[i]["segmentation"], masks[j]["segmentation"])
                #union = np.logical_or(masks[i]["segmentation"], masks[j]["segmentation"]) 
                # we do not use union here, we use the least area large instead
                union = min(np.sum(masks[i]["segmentation"]),np.sum(masks[j]["segmentation"]))
                overlap_area = np.sum(intersection) / union
                overlap_areas[i, j] = overlap_area
                overlap_areas[j, i] = overlap_area

    # Find indices of masks to keep
    keep_indices = set(range(len(masks)))
    for s in small_indices:
        keep_indices.discard(s)
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            if i not in small_indices and j not in small_indices:
                if overlap_areas[i, j] > min_iou:
                    if np.sum(masks[i]["segmentation"]) >= np.sum(masks[j]["segmentation"]):
                        keep_indices.discard(j)
                    else:
                        keep_indices.discard(i)

    # Keep only the masks at the selected indices
    final_masks = [masks[i] for i in sorted(list(keep_indices))]
    areas = [area_large[i] for i in sorted(list(keep_indices))]
    return final_masks,sorted(list(keep_indices)), areas


def are_near_synonyms(phrase1, phrase2):
    # Tokenize phrases
    tokens1 = word_tokenize(phrase1)
    tokens2 = word_tokenize(phrase2)

    # Extract nouns from each phrase
    nouns1 = [token for token, pos in nltk.pos_tag(tokens1) if pos.startswith('N')]
    nouns2 = [token for token, pos in nltk.pos_tag(tokens2) if pos.startswith('N')]

    # Check if nouns are present and compare their similarity
    if nouns1 and nouns2:
        synsets1 = wordnet.synsets(nouns1[0])
        synsets2 = wordnet.synsets(nouns2[0])
        """
        for synset1 in synsets1:
            for synset2 in synsets2:
                similarity = synset1.path_similarity(synset2)
                
                #pdb.set_trace()
                if similarity is not None and similarity > 0.9:
                    return True
        """     
        matching_words = 0

        for synset1 in synsets1:
            for synset2 in synsets2:
                similarity = synset1.path_similarity(synset2)
                if similarity is not None and similarity > 0.9:
                    matching_words += 1
                    if matching_words >= 3:
                        return True
        

    return False

def get_head(doc):
        """Return the token that is the head of the dependency parse."""
        for token in doc:
            if token.head.i == token.i:
                return token
        return None

def get_chunks(doc):
    """Return a dictionary mapping sentence indices to their noun chunk."""
    chunks = {}
    for chunk in doc.noun_chunks:
        for idx in range(chunk.start, chunk.end):
            chunks[idx] = chunk
    return chunks

# def parse_sentence_entities(nlp,sent):
#     counts = {}
#     counts["n_0th_noun"] = 0
#     doc = nlp(sent)
#     head = get_head(doc)
#     chunks = get_chunks(doc)
#     if expand_chunks:
#         chunks = expand_chunks(doc, chunks)
#     entity = Entity.extract(head, chunks, heuristics)

#     # If no head noun is found, take the first one.
#     if entity is None and len(list(doc.noun_chunks)) > 0:
#         head = list(doc.noun_chunks)[0]
#         entity = Entity.extract(head.root.head, chunks, heuristics)
#         counts["n_0th_noun"] += 1
#     #pdb.set_trace()
#     return entity, counts, chunks



if __name__ == "__main__":
    
    # cfg
    # config_file = "/home/ljc/SceneGraphGenZeroShotWithGSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"  # change the path of the model config file
    # grounded_checkpoint = "/home/ljc/SceneGraphGenZeroShotWithGSAM/checkpoints/groundingdino_swint_ogc.pth"  # change the path of the model
    # sam_checkpoint = "/home/ljc/SceneGraphGenZeroShotWithGSAM/checkpoints/sam_vit_h_4b8939.pth"
    # text_prompt = "bear"
    # output_dir = "outputs"
    box_threshold = 0.3
    text_threshold = 0.25
    device = "cuda"
    # torch.cuda.set_device(device)
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    spacy.tokens.Token.set_extension('wordnet', default=None, force=True)

    # Weight_S_p = 1.5
    Weight_S_v = 1
    Weight_S_n = -0.7

    # S_p_global = 0.5
    # S_p_v = 0.75

    Weight_S_g = 0
    Weight_S_l = 0.2
    

    # is_baseline = True
    is_add_global_feature = True
    # 两种情况，第一种是裁剪，第二种是不加S_p ["crop", "none"]
    add_S_p = "none"
    # 两种情况，第一种是整张图，第二种是裁剪， S_v最基本，["image", "crop", "none"]
    add_S_v = "crop"
    # 两种情况, 第一种情况是加上负样本的文本 ["text", "none"]
    add_S_n = "text"
    # 两种情况，第一种是用高斯模糊加上global-local中global的部分，第二种是加crop，第三种是不加global的部分
    # ["blur", "blur_crop", "none"]
    add_global_feature = "blur_crop"
    # 两种情况，加不加空间微调
    is_add_spatial = True
    # 加不加global的llm
    is_add_s_g = False
    # 加不加local的llm
    is_add_s_l = True

   
    # initialize SAM
    # sam = build_sam(checkpoint=sam_checkpoint)
    # sam.to(device=device)

    # mask_generator = SamAutomaticMaskGenerator(sam,points_per_side=8,
    #                  pred_iou_thresh=0.7,
    #                  stability_score_thresh=0.7,
    #                  crop_n_layers=0,
    #                  crop_n_points_downscale_factor=1,
    #                  min_mask_region_area=800,)


    # vit_b_clip_state_dict = torch.jit.load("/home/ljc/SceneGraphGenZeroShotWithGSAM/checkpoints/ViT-B-32.pt",map_location="cpu").eval()
    # vit_b_clip_model = build_model(vit_b_clip_state_dict.state_dict(),77).float().cuda()
    # vit_b_clip_model.eval()

    # clip_state_dict = torch.jit.load("/home/ljc/SceneGraphGenZeroShotWithGSAM/checkpoints/ViT-L-14-336px.pt",map_location="cpu").eval()
    
    # clip_state_dict = torch.jit.load("/home/ljc/SceneGraphGenZeroShotWithGSAM/checkpoints/ViT-B-32.pt",map_location="cpu").eval()
    # # clip_state_dict = torch.jit.load("/home/ljc/SceneGraphGenZeroShotWithGSAM/checkpoints/ViT-L-14.pt",map_location="cpu").eval()
    # clip_model = build_model(clip_state_dict.state_dict(),77).float().cuda()
    # clip_model.eval()
    # preprocess = _transform(clip_model.visual.input_resolution)

    args = get_parser()
    Weight_S_l = args.Weight_S_l
    Weight_S_n = args.Weight_S_n * -1


    # build dataset & dataloader
    if dataset_name == "DuMoGa":
        test_data = DuMoGaTestDataset(data_dir=args.data_dir,
                                split=args.test_split,
                                #mode='test',
                                input_size=args.input_size,
                                word_length=args.word_len,
                                coco_path="/home/ljc/datasets/coco")
    elif dataset_name == "reasonSeg":
        test_data = resonSegTestDataset(data_dir=args.data_dir,
                                split=args.test_split,
                                #mode='test',
                                input_size=args.input_size,
                                word_length=args.word_len)
    else:
        raise Exception("数据集未实现")
    
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              pin_memory=True)
    
    for mode in ["local", "global"]:
        if mode == "local":
            template = """Given a image and the corresponding referring expression "{}", the entity referred by the referring expression is unique in the image. Please generate a caption with local concept to describe the referent object according to the referring expression. The format is "a photo of  <object>(attribute)" """
        else:
            template = """Given a image and the corresponding referring expression "{}", the entity referred by the referring expression is unique in the image. Please generate a caption to describe the referent object and its surrounding entities according to the referring expression. The format is "a photo of  <object> surrounded by (entities)" """


        with open(f'/home/ljc/LLM/datas/preprocess_llm/{dataset_name}_{args.test_split}_{mode}.jsonl','w+') as f:
            for i, data in enumerate(tqdm(test_loader)):
                image, param = data

                image_name = param['file_name'][0]
                sentence_raw = param['sents']
                sentence_ids = param['sents_id']

                # print(image_name, sentence_raw, sentence_ids)
                # exit()

                for sentence, sentence_id in zip(sentence_raw, sentence_ids):
                    instance = {"image":image_name, "text": template.format(sentence[0]), "category": "detail", "question_id":sentence_id[0]}
                    instance_str = json.dumps(instance)
                    f.write(instance_str+"\n")