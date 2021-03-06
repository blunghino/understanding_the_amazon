import os.path
from PIL import Image

import tifffile
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torch
from torch import np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from balance_batch_dataloader import BalanceSampler, BalanceDataLoader
from augment_data import random_flip_rotation_pil


## constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResnetTrainDataset(Dataset):
    """
    class to load Amazon satellite data into pytorch for pretrained resnet
    """
    def __init__(self, csv_path, img_path, dtype, img_ext='.jpg',
                 channel_means=None, channel_stds=None):

        self.img_path = img_path
        self.dtype = dtype
        self.img_ext = img_ext

        df = pd.read_csv(csv_path)

        self.mlb = MultiLabelBinarizer()

        if channel_means is None:
            channel_means = IMAGENET_MEAN
        if channel_stds is None:
            channel_stds = IMAGENET_STD

        ## add all img transforms to this list
        transform_list = [
            transforms.RandomSizedCrop(224),
            random_flip_rotation_pil,
            transforms.ToTensor(),
            transforms.Normalize(mean=channel_means, std=channel_stds)
        ]
        self.transforms = transforms.Compose(transform_list)

        ## the paths to the images
        self.X_train = df['image_name']
        self.y_train = self.mlb.fit_transform(df['tags'].str.split()).astype(np.float32)

    def __getitem__(self, index):
        """
        return X_train image and y_train index
        """
        img_str = self.X_train[index] + self.img_ext
        load_path = os.path.join(self.img_path, img_str)

        if self.img_ext == '.jpg':
            img = Image.open(load_path)
            img = img.convert('RGB')
            img = self.transforms(img)

        elif self.img_ext == '.npy':
            img = np.load(load_path)
            img=np.array(img,dtype=np.int32)
            img=img.reshape((7,256,256))

        label = torch.from_numpy(self.y_train[index]).type(self.dtype)
        return img, label

    def __len__(self):
        return len(self.X_train.index)


class ResnetTestDataset(Dataset):
    """
    class to load test data for Resnet into pytorch
    """
    def __init__(self, csv_path, img_path, dtype, img_ext='.jpg',
                 channel_means=None, channel_stds=None):

        self.img_path = img_path
        self.dtype = dtype
        self.img_ext = img_ext

        df = pd.read_csv(csv_path)

        self.mlb = MultiLabelBinarizer()

        if channel_means is None:
            channel_means = IMAGENET_MEAN
        if channel_stds is None:
            channel_stds = IMAGENET_STD

        ## add all img transforms to this list
        transform_list = [
            transforms.Scale(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=channel_means, std=channel_stds)
        ]
        self.transforms = transforms.Compose(transform_list)

        ## the paths to the images
        self.X_train = df['image_name']

    def __getitem__(self, index):
        """
        return X_train image and y_train index
        """
        img_str = self.X_train[index] + self.img_ext
        load_path = os.path.join(self.img_path, img_str)

        ## branching for different backends
        if self.img_ext == '.jpg':
            img = Image.open(load_path)
            img = img.convert('RGB')
            img = self.transforms(img)

        elif self.img_ext == '.npy':
            img = np.load(load_path)
            img=np.array(img,dtype=np.int32)
            img=img.reshape((7,256,256))

        return img, self.X_train[index]

    def __len__(self):
        return len(self.X_train.index)


class ResnetOptimizeDataset(ResnetTrainDataset):
    """
    class for optimizing weights and thresholds post training
    """
    def __init__(self, csv_path, img_path, dtype, img_ext='.jpg',
                 channel_means=None, channel_stds=None):

        self.img_path = img_path
        self.dtype = dtype
        self.img_ext = img_ext

        df = pd.read_csv(csv_path)

        self.mlb = MultiLabelBinarizer()

        if channel_means is None:
            channel_means = IMAGENET_MEAN
        if channel_stds is None:
            channel_stds = IMAGENET_STD

        ## add all img transforms to this list
        transform_list = [
            transforms.Scale(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=channel_means, std=channel_stds)
        ]
        self.transforms = transforms.Compose(transform_list)

        ## the paths to the images
        self.X_train = df['image_name']
        self.y_train = self.mlb.fit_transform(df['tags'].str.split()).astype(np.float32)



class AmazonDataset(Dataset):
    """
    class to conform data to pytorch API
    """
    def __init__(self, csv_path, img_path, img_ext, dtype,
                 transform_list=[], three_band=False,
                 channel_means=None, channel_stds=None, use_flips=True):

        self.img_path = img_path
        self.img_ext = img_ext
        self.dtype = dtype
        self.three_band = three_band

        df = pd.read_csv(csv_path)

        self.mlb = MultiLabelBinarizer()
        ## prepend other img transforms to this list
        if use_flips:
            transform_list += [random_flip_rotation_pil]

        transform_list += [transforms.ToTensor()]
        if channel_means is not None and channel_stds is not None:
            transform_list += [transforms.Normalize(mean=channel_means,
                                                    std=channel_stds)]
        self.transforms = transforms.Compose(transform_list)
        ## the paths to the images
        self.X_train = df['image_name']
        self.y_train = self.mlb.fit_transform(df['tags'].str.split()).astype(np.float32)



    def __getitem__(self, index):
        """
        return X_train image and y_train index
        """
        img_str = self.X_train[index] + self.img_ext
        load_path = os.path.join(self.img_path, img_str)
        ## branching for different backends
        if self.img_ext == '.jpg':
            img = Image.open(load_path)
            ## convert to three color bands (eg for using with pretrained model)
            if self.three_band:
                img = img.convert('RGB')
        ## tifffile
        elif self.img_ext == '.tif':
            img = tifffile.imread(load_path)
            img = np.asarray(img, dtype=np.int32)

        img = self.transforms(img)
        label = torch.from_numpy(self.y_train[index]).type(self.dtype)
        return img, label

    def __len__(self):
        return len(self.X_train.index)

class AmazonTestDataset(Dataset):
    """
    class to conform data to pytorch API
    """
    def __init__(self, csv_path, img_path, img_ext, dtype,
                 transform_list=[], three_band=False,
                 channel_means=None, channel_stds=None):

        self.img_path = img_path
        self.img_ext = img_ext
        self.dtype = dtype
        self.three_band = three_band

        df = pd.read_csv(csv_path)

        ## prepend other img transforms to this list
        transform_list += [transforms.ToTensor()]
        if channel_means is not None and channel_stds is not None:
            transform_list += [transforms.Normalize(mean=channel_means,
                                                    std=channel_stds)]
        self.transforms = transforms.Compose(transform_list)
        ## the paths to the images
        self.X_train = df['image_name']

    def __getitem__(self, index):
        """
        return X_train image and y_train index
        """
        img_str = self.X_train[index] + self.img_ext
        load_path = os.path.join(self.img_path, img_str)
        ## branching for different backends
        if self.img_ext == '.jpg':
            img = Image.open(load_path)
            ## convert to three color bands (eg for using with pretrained model)
            if self.three_band:
                img = img.convert('RGB')
        ## tifffile
        elif self.img_ext == '.tif':
            img = tifffile.imread(load_path)
            img = np.asarray(img, dtype=np.int32)

        img = self.transforms(img)
        return img,self.X_train[index]

    def __len__(self):
        return len(self.X_train.index)


def generate_train_val_dataloader(dataset, batch_size, num_workers,
                                  shuffle=True, split=0.9, use_fraction_of_data=1.):
    """
    return two Dataloaders split into training and validation
    `split` sets the train/val split fraction (0.9 is 90 % training data)
    u
    """
    ## this is a testing feature to make epochs go faster, uses only some of the available data
    if use_fraction_of_data < 1.:
        n_samples = int(use_fraction_of_data * len(dataset))
    else:
        n_samples = len(dataset)
    inds = np.arange(n_samples)
    train_inds, val_inds = train_test_split(inds, test_size=1-split, train_size=split)

    train_loader = DataLoader(
        dataset,
        sampler=SubsetRandomSampler(train_inds),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset,
        sampler=SubsetRandomSampler(val_inds),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return train_loader, val_loader


def triple_train_val_dataloaders(datasets, batch_size, num_workers,
                                 shuffle=True, split=0.9,
                                 use_fraction_of_data=1.):
    """
    generate three training and three validation dataloaders
    to train triple resnet
    """
    ## this is a testing feature to make epochs go faster, uses only some of the available data
    if use_fraction_of_data < 1.:
        n_samples = int(use_fraction_of_data * len(datasets[0]))
    else:
        n_samples = len(datasets[0])
    inds = np.arange(n_samples)
    train_inds, val_inds = train_test_split(inds, test_size=1-split, train_size=split)

    train_loaders = []
    val_loaders = []

    for dset in datasets:
        train_loaders.append(DataLoader(
            dset,
            sampler=SubsetRandomSampler(train_inds),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        ))
        val_loaders.append(DataLoader(
            dset,
            sampler=SubsetRandomSampler(val_inds),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        ))

    return train_loaders, val_loaders

def triple_train_val_balance_dataloaders(datasets, batch_size, num_workers,
                                         shuffle=True, split=0.9,
                                         use_fraction_of_data=1.):
    """
    generate three training and three validation dataloaders
    to train triple resnet
    """
    n_samples = len(datasets[0])
    ## set up train val split
    inds = np.arange(n_samples)
    train_inds, val_inds = train_test_split(inds, test_size=1-split, train_size=split)
    ## logical indexing to use with BalanceSampler
    log_train_inds = np.zeros(n_samples)
    log_train_inds[train_inds] = 1
    log_val_inds = np.zeros(n_samples)
    log_val_inds[val_inds] = 1
    ## reduce the size of your dataset (use for testing only)
    if use_fraction_of_data < 1:
        train_idx = int(np.ceil(use_fraction_of_data * split * n_samples))
        val_idx = int(np.ceil(use_fraction_of_data * (1-split) * n_samples))
        log_train_inds[train_idx:] = 0
        log_val_inds[val_idx:] = 0

    train_loaders = []
    val_loaders = []

    for dset in datasets:
        train_loaders.append(BalanceDataLoader(
            dset,
            sampler=BalanceSampler(dset, log_train_inds),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        ))
        val_loaders.append(BalanceDataLoader(
            dset,
            sampler=BalanceSampler(dset, log_val_inds),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        ))

    return train_loaders, val_loaders

if __name__ == '__main__':

    from balance_batch_dataloader import *

    csv_path = 'data/train_v2.csv'
    img_path = 'data/train-jpg'
    img_ext = '.jpg'
    dtype = torch.FloatTensor
    training_dataset = ResnetOptimizeDataset(csv_path, img_path, dtype)
    inds = np.arange(10000)
    logical_inds = np.zeros(len(training_dataset))
    logical_inds[inds] = 1
    bbs = BalanceSampler(training_dataset, logical_inds)
    print(len(bbs))
    train_loader = BalanceDataLoader(training_dataset, sampler=bbs,
                                               batch_size=32, num_workers=1)
    for t, (x, y) in enumerate(train_loader):
        col_sum = y.sum(dim=0).numpy().flatten()
        print(col_sum > 0)
        break
