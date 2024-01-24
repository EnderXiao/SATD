import os
import glob
from tqdm import tqdm
import cv2
import pickle as pkl
import torchvision.transforms as tf

from typing import List

import numpy as np


class ScaleToLimitRange:
    def __init__(self, w_lo: int, w_hi: int, h_lo: int, h_hi: int) -> None:
        assert w_lo <= w_hi and h_lo <= h_hi
        self.w_lo = w_lo
        self.w_hi = w_hi
        self.h_lo = h_lo
        self.h_hi = h_hi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        r = h / w
        lo_r = self.h_lo / self.w_hi
        hi_r = self.h_hi / self.w_lo
        assert lo_r <= h / w <= hi_r, f"img ratio h:w {r} not in range [{lo_r}, {hi_r}]"

        scale_r = min(self.h_hi / h, self.w_hi / w)
        if scale_r < 1.0:
            # one of h or w highr that hi, so scale down
            img = cv2.resize(
                img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR
            )
            return img

        scale_r = max(self.h_lo / h, self.w_lo / w)
        if scale_r > 1.0:
            # one of h or w lower that lo, so scale up
            img = cv2.resize(
                img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR
            )
            return img

        # in the rectangle, do not scale
        assert self.h_lo <= h <= self.h_hi and self.w_lo <= w <= self.w_hi
        return img


class ScaleAugmentation:
    def __init__(self, lo: float, hi: float) -> None:
        assert lo <= hi
        self.lo = lo
        self.hi = hi

    def __call__(self, img: np.ndarray) -> np.ndarray:
        k = np.random.uniform(self.lo, self.hi)
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        return img


MAX_SIZE = 32e4  # change here accroading to your GPU memory

K_MIN = 0.7
K_MAX = 1.4

H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024

def gen_pkl(state, scale_aug=False):
    # trans_list =[]
    # if state == 'train' and scale_aug:
    #     trans_list.append(ScaleAugmentation(K_MIN, K_MAX))
    # trans_list += [
    #     ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI),
    # ]
    # transform = tf.Compose(trans_list)

    image_path = 'train/img'
    image_out = 'train_image.pkl'
    laebl_path = 'train_hyb'
    label_out = 'train_label.pkl'
    ignore_list = []

    if state == 'test':
        image_path = 'test/img'
        image_out = 'test_image.pkl'
        laebl_path = 'test_hyb'
        label_out = 'test_label.pkl'

    images = glob.glob(os.path.join(image_path, '*.bmp'))
    image_dict = {}

    for item in tqdm(images):

        img = cv2.imread(item)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fname = os.path.basename(item).replace('.bmp', '')
        # if scale_aug:
        #     img = transform(img)
        if img.shape[0] * img.shape[1] > MAX_SIZE:
            ignore_list.append(fname)
            print(f"image: {fname} size: {img.shape[0]} x {img.shape[1]} =  bigger than {MAX_SIZE}, ignore")
            continue
        # print(os.path.basename(item).replace('_0.bmp', ''))
        # image_dict[os.path.basename(item).replace('_0.bmp', '')] = img
        image_dict[fname] = img

    with open(image_out, 'wb') as f:
        pkl.dump(image_dict, f)

    labels = glob.glob(os.path.join(laebl_path, '*.txt'))
    label_dict = {}

    for item in tqdm(labels):
        with open(item) as f:
            lines = f.readlines()
        fname = os.path.basename(item).replace('.txt', '')
        if fname in ignore_list:
            print(f"image: {fname}, ignore")
            continue
        label_dict[os.path.basename(item).replace('.txt', '')] = lines

    with open(label_out, 'wb') as f:
        pkl.dump(label_dict, f)


if __name__ == '__main__':
    gen_pkl('train', False)
    gen_pkl('test')
