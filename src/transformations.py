import numbers
import numpy as np
from PIL import Image, ImageOps
import torch
from numpy import random


class RandomHorizontallyFlip(object):
    """
    Data augmentation class for flipping an image and its ground truth with 0.5 probability
    """
    def __call__(self, *imgs):
        imgs = [Image.fromarray(img) if not(issubclass(type(img), Image.Image)) else img for img in imgs]

        if np.random.random() < 0.5:
            return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]


class RandomGammaCorrection(object):
    """
    Data augmentation class for applying Gamma Correction with the gamma parameter sampled from a Beta distribution
    """
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, img):
        gamma = np.random.beta(self.alpha, self.beta)
        return np.power(img, 1/gamma)


class Scale(object):
    """
    Scales an array by a given factor
    """
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return torch.from_numpy(np.array(img)) * self.factor


class DeNormalize(object):
    """
    Denormalize an array given mean and standard deviation
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor