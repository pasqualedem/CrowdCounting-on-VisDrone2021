import numpy as np
from tqdm import tqdm
from visdrone2021 import VisDrone2021Dataset, make_dataframe as make_2021


def calculate():
    """
    Calculate the mean and the standard deviation of a dataset
    """
    dft = make_2021('../../datasets/VisDrone2021/Train')
    dfv = make_2021('../../datasets/VisDrone2021/Val')
    dst = VisDrone2021Dataset(dft, train=False)
    dsv = VisDrone2021Dataset(dfv, train=False)
    length = len(dst) + len(dsv)
    means = 0
    stds = 0
    i = 0
    try:
        for img, den in dst:
            means += np.mean((img / 255).reshape(img.shape[0] * img.shape[1], img.shape[2]), axis=0)
            stds += np.std((img / 255).reshape(img.shape[0] * img.shape[1], img.shape[2]), axis=0)
            i += 1
    finally:
        try:
            for img, den in dsv:
                means += np.mean((img / 255).reshape(img.shape[0] * img.shape[1], img.shape[2]), axis=0)
                stds += np.std((img / 255).reshape(img.shape[0] * img.shape[1], img.shape[2]), axis=0)
                i += 1
        finally:
            print(means / length)
            print(stds / length)
            print(i)


if __name__ == '__main__':
    calculate()
