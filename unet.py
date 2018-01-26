from __future__ import division, print_function

import glob as _glob
import os

import keras.callbacks as _callbacks
import keras.layers as _layers
import keras.models as _models
import keras.optimizers as _optimizers
import numpy as _np
import preprocess as _preprocess
from sklearn.utils import shuffle as _shuffle


def _ConvDown(input_tensor, filters):
    """ Convolve twice with kernel size 3 and "same" padding

    :param input_tensor: Tensor of shape (batch_size, length, height, channels)
    :type input_tensor: Tensor
    :param filters: number of convolution filters
    :type filters: int

    :return: Tensor of shape (batch_size, length, height, channels)
    :rtype: Tensor
    """
    out = _layers.Conv2D(filters, 3, padding="same", activation="elu",
                         kernel_initializer="he_normal")(input_tensor)
    return _layers.Conv2D(filters, 3, padding="same", activation="elu",
                          kernel_initializer="he_normal")(out)


def _ConvUp(input_tensor, filters):
    """ Upsample by a factor 2x and then convolve

    :param input_tensor: Tensor of shape (length, height, channels)
    :type input_tensor: Tensor
    :param filters: number of convolution filters
    :type filters: int

    :return: Tensor of shape (2*length, 2*height, channels)
    :rtype: Tensor
    """
    out = _layers.UpSampling2D(size=(2, 2))(input_tensor)
    return _layers.Conv2D(filters, 2, padding="same", activation="elu",
                          kernel_initializer="he_normal")(out)


def UNet(img_size=(512, 512)):
    """
    U{https://arxiv.org/pdf/1505.04597.pdf}

    :param img_size: (length, height) of input images
    :type img_size: tuple

    :return: UNet Keras model
    :rtype: Model
    """
    
    img_rows, img_cols = img_size
    inputs = _layers.Input((img_rows, img_cols, 1))
    
    conv1 = _ConvDown(inputs, 64)
    pool1 = _layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = _ConvDown(pool1, 128)
    pool2 = _layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = _ConvDown(pool2, 256)
    pool3 = _layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = _ConvDown(pool3, 512)
    drop4 = _layers.Dropout(0.5)(conv4)
    pool4 = _layers.MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = _ConvDown(pool4, 1024)
    drop5 = _layers.Dropout(0.5)(conv5)
    
    up6 = _ConvUp(drop5, 512)
    merge6 = _layers.concatenate([drop4, up6])
    conv6 = _ConvDown(merge6, 512)
    
    up7 = _ConvUp(conv6, 256)
    merge7 = _layers.concatenate([conv3, up7])
    conv7 = _ConvDown(merge7, 256)
    
    up8 = _ConvUp(conv7, 128)
    merge8 = _layers.concatenate([conv2, up8])
    conv8 = _ConvDown(merge8, 128)
    
    up9 = _ConvUp(conv8, 64)
    out = _layers.concatenate([conv1, up9])
    out = _ConvDown(out, 64)
    
    out = _layers.Conv2D(2, 3, padding="same", activation="elu",
                         kernel_initializer="he_normal")(out)
    
    out = _layers.Conv2D(1, 1, activation="sigmoid")(out)
    
    model = _models.Model(input=inputs, output=out)
    
    model.compile(optimizer=_optimizers.Adam(lr=3e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    
    return model


def SampleGenerator(images, labels, batch_size=1, random=True, augment=None):
    """ Infinite generator of batches of (input image, labelled image) """
    num_samples = len(images)
    while True:
        # Shuffle the data to avoid looping over in the same order every time
        if random:
            images, labels = _shuffle(images, labels)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = list(range(offset, offset + batch_size))
            
            x_batch = []
            y_batch = []
            
            for idx in batch_samples:
                x = images[idx]
                y = labels[idx]
                if augment is not None:
                    x, y = augment(x, y)
                
                x_batch.append(x)
                y_batch.append(y)
            
            yield _np.array(x_batch), _np.array(y_batch)


def load_img(fname):
    """ Load NumPy array representing B/W image,
    convert it in the range [-1, 1],
    extend it in 3rd dimension (color channels)"""
    img = _np.load(fname).astype("float")
    img = 2 * ((img / 255) - 0.5)
    return img[:, :, None]


def load_label(fname):
    """ Load NumPy array representing B/W image,
    convert it in the range [0, 1] (probability),
    extend it in 3rd dimension (color channels)"""
    img = _np.load(fname).astype("float")
    img /= 255
    return img[:, :, None]


if __name__ == "__main__":
    img_size = 384

    from preprocess_imgs import  make_inputs_from_imgs

    if not os.path.exists('./data'):
        os.mkdir('./data')
        os.mkdir('./data/origs')
        os.mkdir('./data/labels')

    # Create input arrays from folder of images and masks
    make_inputs_from_imgs(img_size, in_folder='../openfriday_nn', out_folder='./data/', extension='png')
    
    img_names = sorted(_glob.glob('./data/origs/*.npy'))
    label_names = sorted(_glob.glob('./data/labels/*.npy'))
    assert len(img_names) == len(label_names)

    X = [load_img(name) for name in img_names]
    Y = [load_label(name) for name in label_names]
    
    model = UNet(img_size=(img_size, img_size))
    print(model.summary())
    
    img_gen = _preprocess.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    ).random_transform_covariant
    
    train_generator = SampleGenerator(X, Y, batch_size=4, random=True,
                                      augment=img_gen)
    
    callbacks = [
        _callbacks.EarlyStopping(patience=50, monitor="loss"),
        _callbacks.ModelCheckpoint("unet.{epoch:03d}.{loss:.5f}.h5",
                                   monitor="loss", save_best_only=True),
    ]
    
    model.fit_generator(
        train_generator,
        4 * len(X),
        epochs=1000,
        verbose=1,
        callbacks=callbacks
    )
