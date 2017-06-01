import tifffile
import pandas as pd
import numpy as np
from PIL import Image
import cv2

csv_path="train_v2.csv"
df = pd.read_csv(csv_path)
file_names=df['image_name']
img_path="train-tif-v2/"
output_path="train_inf/"
for j,i in enumerate(file_names):
    img = tifffile.imread(img_path+i+".tif")
    img = np.asarray(img, dtype=np.uint8)
    img=img[:,:,3]
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    cv2.imwrite(output_path+i+".jpg",img)
