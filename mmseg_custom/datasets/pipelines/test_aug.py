from functools import partial

import cv2
from mmseg.datasets.builder import PIPELINES
import imgaug.augmenters as iaa


def resize_and_resize(image, ratio):
    h, w = image.shape[:2]
    h_, w_ = int(h * ratio), int(w * ratio)
    img_resized = cv2.resize(image, dsize=(w_, h_), interpolation=cv2.INTER_CUBIC)
    img_ = cv2.resize(img_resized, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
    return img_


def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, ksize=(kernel_size, kernel_size), sigmaX=0)


@PIPELINES.register_module(force=True)
class TestAug(object):

    def __init__(self, aug_type, aug_arg):
        self.aug_type = aug_type
        self.aug_arg = aug_arg
        if aug_type == 'Resize':
            self.aug = partial(resize_and_resize, ratio=aug_arg)
        elif aug_type == 'GaussianBlur':
            self.aug = partial(gaussian_blur, kernel_size=aug_arg)
        elif aug_type == 'GaussianNoise':
            self.aug = iaa.AdditiveGaussianNoise(scale=aug_arg)
        elif aug_type == 'JPEGCompression':
            self.aug = iaa.JpegCompression(compression=100 - aug_arg)

    # noinspection PyArgumentList
    def __call__(self, results):
        assert 'img' in results, f'img should in results, but got {results}'
        img = results['img']
        img_aug = self.aug(image=img)
        results['img'] = img_aug
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}, type={self.aug_type}, arg={self.aug_arg}'
