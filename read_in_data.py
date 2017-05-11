import os.path
from PIL import Image 

import pandas as pd 
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch import np 
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class AmazonDataset(Dataset):
    """
    class to conform data to pytorch API
    """
    def __init__(self, csv_path, img_path, dtype, img_ext='.jpg'):
    
        self.img_path = img_path
        self.img_ext = img_ext
        self.dtype = dtype
        
        df = pd.read_csv(csv_path)
        
        self.mlb = MultiLabelBinarizer()
        ## prepend other img transforms to this list
        self.transforms = transforms.Compose([transforms.ToTensor()])
        ## the paths to the images
        self.X_train = df['image_name']
        self.y_train = self.mlb.fit_transform(df['tags'].str.split()).astype(np.float32)

    def __getitem__(self, index):
        """
        return X_train image and y_train index
        """
        img_str = self.X_train[index] + self.img_ext
        img = Image.open(os.path.join(self.img_path, img_str))  
        img = self.transforms(img)
        label = torch.from_numpy(self.y_train[index]).type(self.dtype)
        return img, label

    def __len__(self):
        return len(self.X_train.index)


if __name__ == '__main__':
    csv_path = 'data/train.csv'
    img_path = 'data/train-jpg'
    dtype = torch.FloatTensor
    training_dataset = AmazonDataset(csv_path, img_path, dtype)
    train_loader = DataLoader(
        training_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4 # 1 for CUDA
        # pin_memory=True # CUDA only
    )
