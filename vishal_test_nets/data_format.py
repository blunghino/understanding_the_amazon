import pandas as pd
import os
from shutil import copyfile

csv_path="/Users/vishalsubbiah/Documents/Stanford/3rd_quarter/231N_project/understanding_the_amazon/data/train_v2.csv"
base_path="/Users/vishalsubbiah/Documents/Stanford/3rd_quarter/231N_project/understanding_the_amazon/data/Final/Train_jpg_sample/"

df = pd.read_csv(csv_path)


tags=df['tags']
tags=tags.str.split()

names=df['image_name']

src_dir="/Users/vishalsubbiah/Documents/Stanford/3rd_quarter/231N_project/understanding_the_amazon/data/train-jpg/"
dest_dir="/Users/vishalsubbiah/Documents/Stanford/3rd_quarter/231N_project/understanding_the_amazon/data/Final/Train_jpg/"

train_no=len(names)

for i in range(train_no):
    file_name=names.ix[i]
    for j in range(len(tags.ix[i])):
        directory=base_path+str(tags.ix[i][j])
        if(not os.path.exists(directory)):
            os.makedirs(directory)
        copyfile(src_dir+str(file_name)+".jpg",directory+"/"+str(file_name)+".jpg")
