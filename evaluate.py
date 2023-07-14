import cv2
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import model

test_path = os.path.join(os.curdir, 'Unet', 'dataset', 'stage1_test')

id_ = os.listdir(test_path)
x_test = np.ndarray((0,128,128,3))
y_test = np.ndarray((0,128,128,1))

for n,id in tqdm(enumerate(id_), total=np.size(id_)):
    img1 = cv2.imread(os.path.join(test_path, id, 'images', f'{id}.png'))
    img1 = cv2.resize(img1, (128,128))
    img2 = np.zeros((128,128,1), dtype = np.uint8)
    for i in os.listdir(os.path.join(test_path, id, 'masks')):
        img_temp = cv2.imread(os.path.join(test_path, id, 'masks', i), 0)
        img_temp = cv2.resize(img_temp, (128,128))
        img2 = cv2.add(img2, img_temp)
    
    x_test = np.append(x_test, np.reshape(img1,(1,128,128,3)), axis=0)
    y_test = np.append(y_test, np.reshape(img2,(1,128,128,1)), axis=0)

print('\nData loaded\n------------------------')