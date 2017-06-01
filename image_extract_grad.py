import tifffile
import pandas as pd
import numpy as np
from PIL import Image
import cv2

csv_path="train_v2.csv"
df = pd.read_csv(csv_path)
file_names=df['image_name']
img_path="train-tif-v2/"
output_path="train_grad/"
for j,i in enumerate(file_names):
    new_img=np.zeros((256,256,3))
    img = tifffile.imread(img_path+i+".tif")
    img = np.asarray(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    new_img[:,:,0]=np.asarray(img,dtype=np.uint8)
    new_img[:,:,1]=np.asarray(sobelx,dtype=np.uint8)
    new_img[:,:,2]=np.asarray(sobely,dtype=np.uint8)
    cv2.imwrite(output_path+i+".jpg", new_img)
