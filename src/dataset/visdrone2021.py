import numpy as np
import pandas as pd
import os
import torch
import torchvision
from PIL import Image as pil
import scipy.sparse
import re
from config import cfg
from easydict import EasyDict
import sklearn.model_selection
import transformations as trans

cfg_data = EasyDict()

cfg_data.SIZE = (224, 224)
cfg_data.CHANNELS = 4
cfg_data.FILE_EXTENSION = '.jpg'
cfg_data.GT_FILE_EXTENSION = '.npz'
cfg_data.LOG_PARA = 2550.0

cfg_data.GAMMA_CORRECTION = False
cfg_data.BETA_ALPHA = 4.2
cfg_data.BETA_BETA = 2.4

cfg_data.MEAN = [0.34899642, 0.33474687, 0.35247781, 0.49493573]
cfg_data.STD = [0.15108294, 0.14548512, 0.146422, 0.15939873]


class VisDrone2021Dataset(torch.utils.data.Dataset):
    """
    Dataset subclass for the VisDrone dataset
    """
    def __init__(self, dataframe, train=True, img_transform=True, gt_transform=True):
        """
        Initialize VisDroneDataset object
        @param dataframe: dataframe of columns [id, filename, gt_filename] for loading images and ground truth
        @param train: boolean that specify if dataset is loaded for train (it applies the Random Horizontal Flip)
        @param img_transform: boolean that specify if scaling, normalizing and resizing data
        @param gt_transform: boolean that specify if multiply the GT to the LOG_PARA constant
        """
        self.dataframe = dataframe
        self.train_transforms = None
        self.img_transform = None
        self.gt_transform = None
        if train:
            self.train_transforms = trans.RandomHorizontallyFlip()

        if img_transform:
            # Initialize data transforms
            trans_list = [torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize(mean=cfg_data.MEAN,
                                                           std=cfg_data.STD),
                          torchvision.transforms.Resize(cfg_data.SIZE)
                          ]
            if cfg_data.GAMMA_CORRECTION:
                trans_list.insert(1, trans.RandomGammaCorrection(cfg_data.BETA_ALPHA, cfg_data.BETA_BETA))
            self.img_transform = torchvision.transforms.Compose(trans_list)  # normalize to (-1, 1)

        if gt_transform:
            self.gt_transform = torchvision.transforms.Compose([
                trans.Scale(cfg_data.LOG_PARA)
            ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i):
        """
        Retrieve, load and preprocess an item of the dataset and its ground truth

        @param i: the id in the dataframe of the item
        @return: data and ground truth tensors
        """
        # Obtain the filename and target
        filename = self.dataframe.loc[i]['filename']
        tir_filename = self.dataframe.loc[i]['tir_filename']
        target_filename = self.dataframe.loc[i]['gt_filename']

        # Load the imgs and the ground truth
        with pil.open(filename) as img:
            data = np.array(img)
        with pil.open(tir_filename) as img:
            tir = np.array(img.split()[0])
        data = np.concatenate((data, tir.reshape(*tir.shape[:], 1)), axis=2)
        target = scipy.sparse.load_npz(target_filename).toarray()
        if self.train_transforms:
            data, target = self.train_transforms(data, target)

        if self.img_transform:
            data = self.img_transform(data)

        if self.gt_transform:
            target = self.gt_transform(target)

        return data, target

    def get_targets(self):
        return self.targets


def make_dataframe(folder):
    """
    Given a folder requiring to have subfolders each one containing the the frames and the npz groundtruth,
    builds a dataframe tracking all the dataset files

    @param folder: the path folder from where build the dataframe
    @return: a DataFrame with columns (example folder, example idx, filename, gt filename)
    """
    rgb_folder = os.path.join(folder, 'RGB')
    tir_folder = os.path.join(folder, 'TIR')
    gt_folder = os.path.join(folder, 'RGB')
    folders = os.listdir(rgb_folder)
    dataset = []
    for cur_folder in folders:
        files = os.listdir(os.path.join(rgb_folder, cur_folder))
        for file in files:
            if cfg_data.FILE_EXTENSION in file:
                idx, ext = file.split('.')
                gt = os.path.join(gt_folder, cur_folder,
                                  idx + '_' + re.sub(', |\(|\)|\[|\]', '_', str(cfg_data.SIZE)) + cfg_data.GT_FILE_EXTENSION)
                tir = os.path.join(tir_folder, cur_folder,
                                  idx + 'R' + cfg_data.FILE_EXTENSION)
                dataset.append([idx, os.path.join(rgb_folder, cur_folder, file), tir, gt])
    return pd.DataFrame(dataset, columns=['id', 'filename', 'tir_filename', 'gt_filename'])


def load_test():
    """
    Create a VisDroneDataset object in test mode
    @return: the visdrone testset
    """
    df = make_dataframe('../dataset/VisDrone2020-CC/test')
    ds = VisDrone2021Dataset(df, train=False, gt_transform=False)
    return ds


def load_train_val():
    """
    Create a train and validation DataLoader from the specified folder
    config values are used (VAL_SIZE, VAL_BATCH_SIZE, N_WORKERS)

    @return: the train and validation DataLoader
    """
    train_df = make_dataframe('../datasets/VisDrone2021/Train')
    valid_df = make_dataframe('../datasets/VisDrone2021/Val')

    # df = make_dataframe('../dataset/VisDrone2020-CC/train')
    # # Split the dataframe in train and validation
    # train_df, valid_df = sklearn.model_selection.train_test_split(
    #     df, test_size=cfg.VAL_SIZE, shuffle=True
    # )
    # train_df = train_df.reset_index(drop=True)
    # valid_df = valid_df.reset_index(drop=True)

    train_set = VisDrone2021Dataset(train_df)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.TRAIN_BATCH_SIZE, num_workers=cfg.N_WORKERS, shuffle=True)

    val_set = VisDrone2021Dataset(valid_df)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.VAL_BATCH_SIZE, num_workers=cfg.N_WORKERS, shuffle=True)

    return train_loader, val_loader

