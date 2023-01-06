from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils import data
from torch.utils.data import Dataset
import logging
import cv2
from torch.utils.data import DataLoader
# import transform_cv2 as T 




class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, img_size, trans_func, mask_suffix=''):
        '''
        构建通用语义分割数据集类
        imgs_dir：图像路径
        masks_dir: mask图像路径（单通道）
        img_size：网络输入大小
        trans_func: 数据增强方法
        mask_suffix: mask文件名与image文件名之间的区别
        '''
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.trans_func = trans_func
        self.mask_suffix = mask_suffix
        assert(self.img_size[0] % 16 == 0), "The input width should be a multiple of 16"
        assert(self.img_size[1] % 16 == 0), "The input height should be a multiple of 16"

        self.image_ids = [splitext(file)[0] for file in listdir(self.imgs_dir) if not file.startswith('.')]
        self.mask_ids = [splitext(file)[0] for file in listdir(self.masks_dir) if not file.startswith('.')]

        assert(len(self.image_ids) == len(self.mask_ids)), "The number of image and mask should match"
        logging.info(f'Creating dataset with {len(self.image_ids)} examples')

    def __len__(self):
        return len(self.image_ids)

    @classmethod
    def preprocess(cls, pil_img, img_size):
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        newW, newH = img_size
        # print("newW={} newH={}".format(newW, newH))
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans
        
    def get_image(self, impath, lbpath):
        # BGR -> GRB
        image = cv2.imread(impath)[:, :, ::-1]
        # gray
        mask = cv2.imread(lbpath, 0)
        return image, mask

    def __getitem__(self, i):
        idx = self.image_ids[i]
        # mask 和 image的图像路径
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1,  f'Either no image or multiple images found for the ID {idx}: {img_file}'

        # 读取image和mask数据
        image, mask = self.get_image(img_file[0], mask_file[0])
        # assert img.size == mask.size, f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        # 数据增强
        im_lb = dict(im=image, lb=mask)
        if not self.trans_func is None:
            im_lb = self.trans_func(im_lb)
        return im_lb


if __name__ == "__main__":
    # 自定义预处理
    class Resize(object):
        '''
        size should be a tuple of (W, H)
        '''
        def __init__(self, size=(384, 384)):
            self.size = size

        def __call__(self, im_lb):
            if self.size is None:
                return im_lb

            im, lb = im_lb['im'], im_lb['lb']
            assert im.shape[:2] == lb.shape[:2]

            resize_h, resize_w   = self.size

            im = cv2.resize(im, (resize_w, resize_h))
            lb = cv2.resize(lb, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)

            return dict(im=im, lb=lb)

    imgs_dir = "./data/imgs/"
    masks_dir = "./data/masks/"
    img_size = (640, 960)
    trans_func = T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
            T.Normalize()
        ])
    dataset = BasicDataset(imgs_dir, masks_dir, img_size, trans_func)
    data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            drop_last=False,
        )
    for batch in data_loader:
        image = batch["im"]
        mask = batch["lb"]
