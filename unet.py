import cv2
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import att_model as model

train_path = os.path.join(os.curdir, 'Unet', 'dataset', 'stage1_train')
# test_path = os.path.join(os.curdir, 'Unet', 'dataset', 'stage1_test')

id_ = os.listdir(train_path)
x_train = np.ndarray((0,128,128,3))
y_train = np.ndarray((0,128,128,1))

for n,id in tqdm(enumerate(id_), total=np.size(id_)):
    img1 = cv2.imread(os.path.join(train_path, id, 'images', f'{id}.png'))
    img1 = cv2.resize(img1, (128,128))
    img2 = np.zeros((128,128,1), dtype = np.uint8)
    for i in os.listdir(os.path.join(train_path, id, 'masks')):
        img_temp = cv2.imread(os.path.join(train_path, id, 'masks', i), 0)
        img_temp = cv2.resize(img_temp, (128,128))
        img2 = cv2.add(img2, img_temp)
    
    x_train = np.append(x_train, np.reshape(img1,(1,128,128,3)), axis=0)
    y_train = np.append(y_train, np.reshape(img2,(1,128,128,1)), axis=0)

print('\nData loaded\n------------------------')

model1 = model.createUnet()

model1.fit(x_train,y_train, 32, 10, validation_split = 0.3)
tf.keras.models.save_model(model1, os.path.join(os.curdir, 'Unet', 'models', 'model2.h5'))