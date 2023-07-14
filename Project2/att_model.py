import tensorflow as tf

IMG_WIDTH=128
IMG_HEIGHT=128
IMG_CHANNELS=3
#build the model

def createUnet(summary = True):
    inputs=tf.keras.layers.Input((IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS))

    s=tf.keras.layers.Lambda(lambda x: x/255)(inputs)

    c1=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(s)
    c1=tf.keras.layers.Dropout(0.1)(c1)
    c1=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c1)
    p1=tf.keras.layers.MaxPooling2D((2,2))(c1)

    c2=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p1)
    c2=tf.keras.layers.Dropout(0.1)(c2)
    c2=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c2)
    p2=tf.keras.layers.MaxPooling2D((2,2))(c2)

    c3=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p2)
    c3=tf.keras.layers.Dropout(0.2)(c3)
    c3=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c3)
    p3=tf.keras.layers.MaxPooling2D((2,2))(c3)

    c4=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p3)
    c4=tf.keras.layers.Dropout(0.2)(c4)
    c4=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c4)
    p4=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(c4)

    c5=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(p4)
    c5=tf.keras.layers.Dropout(0.3)(c5)
    c5=tf.keras.layers.Conv2D(256,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c5)

    #expansive path
    l6=tf.keras.layers.Conv2D(128,(1,1),activation='relu',kernel_initializer='he_normal',padding='same')(c5)
    u6=tf.keras.layers.Conv2D(128,(2,2),strides=(2,2),activation='relu',kernel_initializer='he_normal',padding='same')(c4)
    a6=tf.keras.layers.add([l6,u6])
    a6=tf.keras.layers.ReLU()(a6)
    a6=tf.keras.layers.Conv2D(1,(1,1),kernel_initializer='he_normal',padding='same')(a6)
    a6=tf.keras.layers.Activation('sigmoid')(a6)
    # u6=tf.keras.layers.Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(a6)
    u6=tf.keras.layers.UpSampling2D((2,2))(a6)
    u6=tf.keras.layers.Lambda(lambda x, repnum: tf.keras.backend.repeat_elements(x, repnum, axis=3),arguments={'repnum': 128})(u6)
    u6=tf.keras.layers.multiply([u6,c4])
    c6=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u6)
    c6=tf.keras.layers.Dropout(0.2)(c6)
    c6=tf.keras.layers.Conv2D(128,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c6)

    l7=tf.keras.layers.Conv2D(64,(1,1),activation='relu',kernel_initializer='he_normal',padding='same')(c6)
    u7=tf.keras.layers.Conv2D(64,(2,2),strides=(2,2),activation='relu',kernel_initializer='he_normal',padding='same')(c3)
    a7=tf.keras.layers.add([l7,u7])
    a7=tf.keras.layers.ReLU()(a7)
    a7=tf.keras.layers.Conv2D(1,(1,1),kernel_initializer='he_normal',padding='same')(a7)
    a7=tf.keras.layers.Activation('sigmoid')(a7)
    # u7=tf.keras.layers.Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(a7)
    u7=tf.keras.layers.UpSampling2D((2,2))(a7)
    u7=tf.keras.layers.Lambda(lambda x, repnum: tf.keras.backend.repeat_elements(x, repnum, axis=3),arguments={'repnum': 64})(u7)
    u7=tf.keras.layers.multiply([u7,c3])
    c7=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u7)
    c7=tf.keras.layers.Dropout(0.2)(c7)
    c7=tf.keras.layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c7)

    l8=tf.keras.layers.Conv2D(32,(1,1),activation='relu',kernel_initializer='he_normal',padding='same')(c7)
    u8=tf.keras.layers.Conv2D(32,(2,2),strides=(2,2),activation='relu',kernel_initializer='he_normal',padding='same')(c2)
    a8=tf.keras.layers.add([l8,u8])
    a8=tf.keras.layers.ReLU()(a8)
    a8=tf.keras.layers.Conv2D(1,(1,1),kernel_initializer='he_normal',padding='same')(a8)
    a8=tf.keras.layers.Activation('sigmoid')(a8)
    # u8=tf.keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(a8)
    u8=tf.keras.layers.UpSampling2D((2,2))(a8)
    u8=tf.keras.layers.Lambda(lambda x, repnum: tf.keras.backend.repeat_elements(x, repnum, axis=3),arguments={'repnum': 32})(u8)
    u8=tf.keras.layers.multiply([u8,c2])
    c8=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u8)
    c8=tf.keras.layers.Dropout(0.1)(c8)
    c8=tf.keras.layers.Conv2D(32,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c8)

    l9=tf.keras.layers.Conv2D(16,(1,1),activation='relu',kernel_initializer='he_normal',padding='same')(c8)
    u9=tf.keras.layers.Conv2D(16,(2,2),strides=(2,2),activation='relu',kernel_initializer='he_normal',padding='same')(c1)
    a9=tf.keras.layers.add([l9,u9])
    a9=tf.keras.layers.ReLU()(a9)
    a9=tf.keras.layers.Conv2D(1,(1,1),kernel_initializer='he_normal',padding='same')(a9)
    a9=tf.keras.layers.Activation('sigmoid')(a9)
    # u9=tf.keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),padding='same')(a9)
    u9=tf.keras.layers.UpSampling2D((2,2))(a9)
    u9=tf.keras.layers.Lambda(lambda x, repnum: tf.keras.backend.repeat_elements(x, repnum, axis=3),arguments={'repnum': 16})(u9)
    u9=tf.keras.layers.multiply([u9,c1])
    c9=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(u9)
    c9=tf.keras.layers.Dropout(0.1)(c9)
    c9=tf.keras.layers.Conv2D(16,(3,3),activation='relu',kernel_initializer='he_normal',padding='same')(c9)

    outputs=tf.keras.layers.Conv2D(1,(1,1),activation='sigmoid')(c9)

    model=tf.keras.Model(inputs=[inputs],outputs=[outputs])
    # model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy',tf.keras.metrics.BinaryIoU([0,1])])
    model.compile(optimizer='adam',loss=[tf.keras.losses.BinaryCrossentropy(gamma=2, from_logits=True)],metrics=['accuracy',tf.keras.metrics.BinaryIoU([0,1])])
    if summary:
        model.summary()
    return model

def debug():
    createUnet()

if __name__ == '__main__':
    debug()
