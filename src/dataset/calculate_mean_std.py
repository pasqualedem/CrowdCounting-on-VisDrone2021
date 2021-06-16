import numpy as np
from tqdm import tqdm
from dataset.visdrone2020 import VisDrone2020Dataset, make_dataframe as make_2020
from dataset.visdrone2021 import VisDrone2021Dataset, make_dataframe as make_2021


def calculate():
    """
    Calculate the mean and the standard deviation of a dataset
    """
    df = make_2021('../dataset/VisDrone2020-CC/sequences')
    ds = VisDrone2021Dataset(df, train=False)
    length = len(ds)
    means = 0
    stds = 0
    i = 0
    try:
        for img, den in tqdm(ds):
            means += np.mean((img / 255).reshape(img.shape[0] * img.shape[1], img.shape[2]), axis=0)
            stds += np.std((img / 255).reshape(img.shape[0] * img.shape[1], img.shape[2]), axis=0)
            i += 1
    finally:
        print(means / length)
        print(stds / length)
        print(i)


if __name__ == '__main__':
    calculate()
