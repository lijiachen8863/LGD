import json
import numpy as np
import matplotlib.pyplot as plt

def get_keep_id_list():
    with open("/home/ljc/TAS/results/ours.json", "r") as f:
        our_iou_id_dict = json.load(f)
    with open("/home/ljc/TAS/results/clip_only.json", "r") as f:
        clip_iou_id_dict = json.load(f)
    with open("/home/ljc/TAS/results/global_local_sam.json", "r") as f:
        gl_iou_id_dict = json.load(f)
    with open("/home/ljc/TAS/results/tas.json", "r") as f:
        tas_iou_id_dict = json.load(f)
    
    keep_id_list=[]

    for key in our_iou_id_dict.keys():
        if our_iou_id_dict[key] > clip_iou_id_dict[key] and \
                our_iou_id_dict[key] > gl_iou_id_dict[key] and \
                our_iou_id_dict[key] > tas_iou_id_dict[key]:
            keep_id_list.append(key)
    
    return keep_id_list

def wrap_text(text, max_width):
    """
    Wrap text into multiple lines, preserving existing newlines.
    
    text: str
    max_width: int, maximum number of characters in a line.
    return: str, text with preserved and automatic line breaks.
    """
    lines = []
    for paragraph in text.split('\n'):  # Preserve existing newlines
        words = paragraph.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_width:
                current_line += (word + " ")
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
    return "\n".join(lines)

def show_mask(mask, ax, random_color=False):
    """
    mask: torch.tensor
    ax: plt

    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
        # color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_mask_image(mask, image, title=None, file_name=None, max_title_width=70):
    """
    mask: torch.tensor
    image: plt.Image
    title(optional): str
    file_name(optional): str. if not be identified, it will be nothing.
    max_title_width(optional): int, maximum width of the title per line.
    
    return plt.gca()
    """
    plt.figure(figsize=(10, 10))
    # plt.figure()
    plt.imshow(image)
    show_mask(mask.cpu(), plt.gca())
    plt.axis('off')

    # Handle title with automatic line wrapping
    if title:
        wrapped_title = wrap_text(title, max_title_width)
        plt.gcf().text(
            0.08, 0.98, wrapped_title, ha='left', va='top', fontsize=18, wrap=True
        )

    # Save the figure if a filename is specified
    if file_name:
        plt.savefig(file_name)
    plt.close()
    return plt.gca()

# import cv2
# from PIL import Image
# import torch
# def preprocess_image(image):
#     """
#     Preprocess the image to ensure it's in numpy format (H, W, C).
#     Supports PIL.Image.Image, numpy.ndarray, and torch.Tensor.
#     """
#     if isinstance(image, Image.Image):  # PIL Image
#         image = np.array(image)
#     elif isinstance(image, torch.Tensor):  # torch.Tensor
#         image = image.cpu().numpy()
#         if image.ndim == 3:  # (C, H, W) to (H, W, C)
#             image = np.transpose(image, (1, 2, 0))
#         elif image.ndim == 2:  # Grayscale (H, W)
#             image = np.expand_dims(image, axis=-1)
#     elif not isinstance(image, np.ndarray):
#         raise ValueError("Unsupported image type.")
    
#     # Ensure image has 3 channels for consistency (grayscale to RGB)
#     if image.ndim == 2:  # Grayscale
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     elif image.shape[2] == 1:  # Single-channel to 3-channel
#         image = np.repeat(image, 3, axis=2)
#     return image

# def preprocess_mask(mask, target_size):
#     """
#     Preprocess the mask to ensure it's a binary mask in numpy format (H, W).
#     Supports numpy.ndarray and torch.Tensor.
#     Resizes the mask to match the target size.
#     """
#     if isinstance(mask, torch.Tensor):  # torch.Tensor
#         mask = mask.cpu().numpy()
#     elif not isinstance(mask, np.ndarray):
#         raise ValueError("Unsupported mask type.")
    
#     # Ensure mask is binary (values 0 or 1)
#     mask = (mask > 0).astype(np.uint8)
    
#     # Resize mask to match target image size
#     mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
#     return mask

# def show_mask_image(mask, image, title=None, file_name=None, random_color=False):
#     """
#     mask: torch.Tensor or numpy.ndarray
#     image: PIL.Image.Image, numpy.ndarray, or torch.Tensor
#     title(optional): str
#     file_name(optional): str. If not provided, it will not save.
#     random_color(optional): bool. Use random color for the mask overlay.
#     """
#     # Preprocess image and mask
#     image = preprocess_image(image)
#     mask = preprocess_mask(mask, image.shape[:2])

#     # Generate overlay color
#     if random_color:
#         color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
#     else:
#         color = np.array([255, 0, 0], dtype=np.uint8)  # RGB: Dodger Blue
    
#     # Create overlay
#     overlay = image.copy()
#     overlay[mask > 0] = (overlay[mask > 0] * 0.4 + color * 0.6).astype(np.uint8)

#     # Combine image and overlay
#     combined = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

#     # Display the image
#     if title:
#         cv2.putText(combined, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#     if file_name:
#         cv2.imwrite(file_name, combined)
#     else:
#         cv2.imwrite("visualization.png", combined)
