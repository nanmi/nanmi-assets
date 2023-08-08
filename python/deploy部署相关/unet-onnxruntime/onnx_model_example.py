import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

def preprocess_image(raw_bgr_image, input_h, input_w):
    """
    description: Convert BGR image to RGB,
                    resize and pad it to target size, normalize to [0,1],
                    transform to NCHW format.
    param:
        input_image_path: str, image path
    return:
        image:  the processed image
        image_raw: the original image
        h: original height
        w: original width
    """
    image_raw = raw_bgr_image
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = input_w / w
    r_h = input_h / h
    if r_h > r_w:
        tw = input_w
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((input_h - th) / 2)
        ty2 = input_h - th - ty1
    else:
        tw = int(r_h * w)
        th = input_h
        tx1 = int((input_w - tw) / 2)
        tx2 = input_w - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
    )
    src_resize_img = image
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)
    return image, cv2.cvtColor(src_resize_img, cv2.COLOR_BGR2RGB)

def preprocess_image_v1(raw_bgr_image, input_h, input_w):
    """
    description: Convert BGR image to RGB,
                    resize and pad it to target size, normalize to [0,1],
                    transform to NCHW format.
    param:
        input_image_path: str, image path
    return:
        image:  the processed image
        image_raw: the original image
        h: original height
        w: original width
    """
    image_raw = raw_bgr_image

    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (input_w, input_h))
    
    src_resize_img = image
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)
    return image, cv2.cvtColor(src_resize_img, cv2.COLOR_BGR2RGB)


def get_color_map_list(num_classes, custom_color=None):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.

    Args:
        num_classes (int): Number of classes.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map


def visualize(image, result, color_map, save_dir=None, weight=0.6):
    """
    Convert predict result to color image, and save added image.

    Args:
        image (str): The path of origin image.
        result (np.ndarray): The predict result of image.
        color_map (list): The color used to save the prediction results.
        save_dir (str): The directory for saving visual image. Default: None.
        weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6

    Returns:
        vis_result (np.ndarray): If `save_dir` is None, return the visualized result.
    """

    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")
    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(result, color_map[:, 0])
    c2 = cv2.LUT(result, color_map[:, 1])
    c3 = cv2.LUT(result, color_map[:, 2])
    pseudo_img = np.dstack((c3, c2, c1))

    im = cv2.imread(image)
    print(im.shape)
    pseudo_img = np.reshape(pseudo_img, (640, 940, 3))
    print(pseudo_img.shape)
    vis_result = cv2.addWeighted(im, weight, pseudo_img, 1 - weight, 0)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_name = os.path.split(image)[-1]
        out_path = os.path.join(save_dir, image_name)
        cv2.imwrite(out_path, vis_result)
    else:
        return vis_result

im_path = 'test.jpg'
image = cv2.imread(im_path)

img, src_resize_img = preprocess_image(image, 640, 940)
cv2.imwrite("./tt1.jpg", src_resize_img)
# Using a quantization ONNX model
onnx_model_path = "./wuhan_v2_1.onnx"
ort_session = onnxruntime.InferenceSession(onnx_model_path)

ort_inputs = {
    "x": img
}


masks_index = ort_session.run(None, ort_inputs)

print(masks_index[0])
print(tuple(masks_index[0].shape))
print(np.all(masks_index == 0))
print(np.max(masks_index[0]))
heatmap = masks_index[0][0]
heatmap = np.array(heatmap, dtype=np.uint8)
thresh, result0 = cv2.threshold(heatmap, 0.5, 255, cv2.THRESH_BINARY)
cv2.imwrite("./mask.jpg", result0)

pred = masks_index
pred = np.squeeze(pred, axis=0).astype('uint8')

# save added image
color_map = get_color_map_list(256, custom_color=None)
added_image = visualize("./tt1.jpg", pred, color_map, weight=0.6)

cv2.imwrite("./ttttttt.jpg", added_image)