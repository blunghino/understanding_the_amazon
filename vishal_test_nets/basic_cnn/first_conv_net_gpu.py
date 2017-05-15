"""
testing the basic setup of a model script using a model with two conv layers
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from training_utils import train
from layers import Flatten
from read_in_data import AmazonDataset


## cpu dtype
dtype = torch.FloatTensor
save_model_path = "model_state_dict.pkl"
csv_path = '../../data/train_v2.csv'
img_path = '../../data/train-jpg'
training_dataset = AmazonDataset(csv_path, img_ext=".jpg",img_path, dtype)
## loader
train_loader = DataLoader(
    training_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=1 # 1 for CUDA
    # pin_memory=True # CUDA only
)
## simple linear model
temp_model=nn.Sequential(
    nn.Conv2d(4, 16, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(16),
    nn.AdaptiveMaxPool2d(128),
    nn.Conv2d(16, 32, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(32),
    nn.AdaptiveMaxPool2d(64),
    Flatten())

temp_model = temp_model.type(dtype)
temp_model.train()
size=0
for t, (x, y) in enumerate(train_loader):
            x_var = Variable(x.type(dtype))
            size=temp_model(x_var).size()
            if(t==0):
                break

model = nn.Sequential(
nn.Conv2d(4, 16, kernel_size=3, stride=1),
nn.ReLU(inplace=True),
nn.BatchNorm2d(16),
nn.AdaptiveMaxPool2d(128),
nn.Conv2d(16, 32, kernel_size=3, stride=1),
nn.ReLU(inplace=True),
nn.BatchNorm2d(32),
nn.AdaptiveMaxPool2d(64),
Flatten(),
nn.Linear(size[1], 1024),
nn.ReLU(inplace=True),
nn.Linear(1024, 17))

model.type(dtype)
model.train()
loss_fn = nn.MultiLabelSoftMarginLoss().type(dtype)
optimizer = optim.Adam(model.parameters(), lr=5e-2)

train(train_loader, model, loss_fn, optimizer, dtype,num_epochs=1, print_every=1)

torch.save(model.state_dict(), save_model_path)
state_dict = torch.load(save_model_path)
model.load_state_dict(state_dict)
