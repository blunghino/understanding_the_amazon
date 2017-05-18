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


class AmazonDataset(Dataset):
    """
    class to conform data to pytorch API
    """
    def __init__(self, csv_path, img_path, img_ext, dtype,
                 transform_list=[], three_band=False):
    
        self.img_path = img_path
        self.img_ext = img_ext
        self.dtype = dtype
        self.three_band = three_band

        df = pd.read_csv(csv_path)
        
        self.mlb = MultiLabelBinarizer()
        ## prepend other img transforms to this list
        transform_list += [transforms.ToTensor()]
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

if __name__ == '__main__':
    csv_path = 'data/train_v2.csv'
    img_path = 'data/train-tif-sample'
    img_ext = '.tif'
    dtype = torch.FloatTensor
    training_dataset = AmazonDataset(csv_path, img_path, img_ext, dtype)
    train_loader = gener
    for t, (x, y) in enumerate(train_loader):
        print(x.size())
        break
