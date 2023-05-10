import torchvision.transforms as T
import numpy as np
import os
import cv2

# Define your custom resize function
def custom_resize(img, size):
    '''
    img: orignal image
    size: tuple (width, height)
    '''
    # Perform your custom resize processing
    h, w, c = img.shape
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = size[0] / w
    r_h = size[1] / h
    if r_h > r_w:
        tw = size[0]
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((size[1] - th) / 2)
        ty2 = size[1] - th - ty1
    else:
        tw = int(r_h * w)
        th = size[1]
        tx1 = int((size[0] - tw) / 2)
        tx2 = size[0] - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
    )
    return image

# Define your custom transformation
class CustomResizeTransform(object):
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):
        # Call your custom resize function with the input image and size
        img = custom_resize(img, self.size)
        # # Perform other transformations as needed
        # img = transforms.ToTensor()(img)
        # img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img


imgs_dir = "./imgs/"
data_dir = "./data/"
img_size = (640, 640)
trans_func = T.Compose([
        CustomResizeTransform(img_size),
        T.ToTensor(),
        T.Normalize(mean = [0.485, 0.456, 0.406], 
                    std = [0.229, 0.224, 0.225])
    ])


imgs_list = os.listdir(imgs_dir)
img_length = len(imgs_list)

for i, img_ in enumerate(imgs_list):
    print(f"\t {i+1:4}/{img_length} - {img_}")
    img_in = cv2.imread(os.path.join(imgs_dir, img_))
    img = trans_func(img_in)
    img = img.unsqueeze(0)
    img_array = img.numpy()
    name = img_[:-4]
    np.save(os.path.join(data_dir, f'{name}.npy'), img_array)


