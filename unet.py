from __future__ import division, print_function

import glob
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D,
			  Dropout, concatenate)
from keras.models import Model
from keras.optimizers import Adam
from sklearn.utils import shuffle as _shuffle

from preprocess import ImageDataGenerator


def _ConvDown(input_tensor, filters):
    out = Conv2D(filters, 3, padding='same', activation='elu',
		 kernel_initializer='he_normal')(input_tensor)
    return Conv2D(filters, 3, padding='same', activation='elu',
                  kernel_initializer='he_normal')(out)


def _ConvUp(input_tensor, filters):
    out = UpSampling2D(size=(2, 2))(input_tensor)
    return Conv2D(filters, 2, padding='same', activation='elu',
		  kernel_initializer='he_normal')(out)


def UNet(img_size=(512, 512)):
    """
    https://arxiv.org/pdf/1505.04597.pdf
    """
    
    img_rows, img_cols = img_size
    inputs = Input((img_rows, img_cols, 1))
    
    conv1 = _ConvDown(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = _ConvDown(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = _ConvDown(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = _ConvDown(pool3, 512)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = _ConvDown(pool4, 1024)
    drop5 = Dropout(0.5)(conv5)
        
    up6 = _ConvUp(drop5, 512)
    merge6 = concatenate([drop4, up6])
    conv6 = _ConvDown(merge6, 512)
        
    up7 = _ConvUp(conv6, 256)
    merge7 = concatenate([conv3, up7])
    conv7 = _ConvDown(merge7, 256)
    
    up8 = _ConvUp(conv7, 128)
    merge8 = concatenate([conv2, up8])
    conv8 = _ConvDown(merge8, 128)

    up9 = _ConvUp(conv8, 64)
    out = concatenate([conv1, up9])
    out = _ConvDown(out, 64)
    
    out = Conv2D(2, 3, padding='same', activation='elu',
                 kernel_initializer='he_normal')(out)
    
    out = Conv2D(1, 1, activation='sigmoid')(out)
    
    model = Model(input=inputs, output=out)
    
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model


def SampleGenerator(X, Y, batch_size=1, random=True, augment=None):
    """
    Infinite generator of batches of (input image, labelled image)
    """
    num_samples = len(X)
    while True:
        # Shuffle the data to avoid looping over in the same order every time
        if random:
            X, Y = _shuffle(X, Y)
            
        for offset in range(0, num_samples, batch_size):
            batch_samples = list(range(offset, offset + batch_size))

            X_batch = []
            y_batch = []
            
            for idx in batch_samples:
                x, y = X[idx], Y[idx]
                if augment:
                    x, y = augment(x, y)
                    
                X_batch.append(x)
                y_batch.append(y)

            yield np.array(X_batch), np.array(y_batch)


def load_img(fname):
    """ Load NumPy array representing B/W image,
    convert it in the range [-1, 1],
    extend it in 3rd dimension """
    img = np.load(fname).astype('float')
    img = 2 * ((img / 255) - 0.5)
    return img[:, :, None]

def load_label(fname):
    """ Load NumPy array representing B/W image,
    convert it in the range [0, 1] (probability),
    extend it in 3rd dimension """
    img = np.load(fname).astype('float')
    img /= 255
    return img[:, :, None]


if __name__ == '__main__':
    
    imgnames = sorted(glob.glob('data/origs/*.npy'))
    labelnames = sorted(glob.glob('data/labels/*.npy'))
    assert len(imgnames) == len(labelnames)
    
    X = [load_img(imname) for imname in imgnames]
    Y = [load_label(imname) for imname in labelnames]

    model = UNet()
    print(model.summary())

    imggen = ImageDataGenerator(
    	rotation_range=20,
    	width_shift_range=0.2,
    	height_shift_range=0.2,
    	shear_range=0.2,
    	zoom_range=0.2,
    	horizontal_flip=True
    )

    train_generator = SampleGenerator(X, Y, batch_size=1, random=True,
                                      augment=imggen.random_transform)

    callbacks = [
     	EarlyStopping(patience=50, monitor='loss'),
    	ModelCheckpoint('unet.{epoch:03d}.{loss:.5f}.h5',
                        monitor='loss', save_best_only=True),
    ]

    model.fit_generator(
        train_generator,
        4 * len(X),
        epochs=1000,
        verbose=1,
        callbacks=callbacks
    )
