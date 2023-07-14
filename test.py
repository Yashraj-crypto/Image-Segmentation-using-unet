import tensorflow as tf
import os
import cv2
import numpy as np
import att_model

def filter(img):
    for i in range(len(img)):
        for j in range(len(img[i])):
            if(img[i][j]>0.5):
                img[i][j]=255
            else:
                img[i][j]=0
    return img
        

model1 = tf.keras.models.load_model(os.path.join(os.curdir, 'Unet', 'models', 'model_bin_focal_loss.h5'))
model2 = tf.keras.models.load_model(os.path.join(os.curdir, 'Unet', 'models', 'model2.h5'))
model3 = tf.keras.models.load_model(os.path.join(os.curdir, 'Unet', 'models', 'model3.h5'))
# model4 = tf.keras.models.load_model(os.path.join(os.curdir, 'Unet', 'models', 'model4.h5'))


img1 = cv2.imread('Unet\\dataset\\stage1_train\\00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e\\images\\00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e.png')
img1 = cv2.resize(img1, (128,128))
img = np.reshape(img1, (1,128,128,3))

# sub_model = tf.keras.Model(inputs = model1.input, outputs = model1.layers[37].output)

out1 = model1.predict(img)
out2 = model2.predict(img)
out3 = model3.predict(img)
# out4 = model4.predict(img)

out1 = filter(np.reshape(out1, (128,128)))
out2 = filter(np.reshape(out2, (128,128)))
out3 = filter(np.reshape(out3, (128,128)))
# out4 = filter(np.reshape(out4, (128,128)))


img2 = np.zeros((128,128,1), dtype = np.uint8)
for i in os.listdir(os.path.join('Unet\\dataset\\stage1_train\\00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e\\', 'masks')):
        img_temp = cv2.imread(os.path.join('Unet\\dataset\\stage1_train\\00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e\\', 'masks', i), 0)
        img_temp = cv2.resize(img_temp, (128,128))
        img2 = cv2.add(img2, img_temp)

# print(out1, np.shape(out1), sep='\n')
# out = np.concatenate((out1, out2, out3, out4, img2), axis=1)
out = np.concatenate((out1,np.ones((128,5))*255,out2,np.ones((128,5))*255,out3,np.ones((128,5))*255,img2), axis=1)
cv2.imshow('out.png', out)
cv2.imwrite('out.png', out)
cv2.waitKey(0)
cv2.destroyAllWindows