import numpy as np
import glob
import matplotlib.pyplot as plt

from sklearn.utils import shuffle as _shuffle

from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from preprocess import ImageDataGenerator



def UNet(img_size=(512, 512)):

	img_rows, img_cols = img_size
	inputs = Input((img_rows, img_cols, 1))

	conv1 = Conv2D(64, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
	conv1 = Conv2D(64, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
	drop4 = Dropout(0.5)(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
	drop5 = Dropout(0.5)(conv5)

	up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
	merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

	up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
	merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

	up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
	merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

	up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
	merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

	model = Model(input=inputs, output=conv10)

	model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

	return model



def generator(X, Y, batch_size=1, random=True, augment=None):

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
	img = np.loadtxt(fname).astype(float)
	img = 2 * ((img / 255.) - 0.5)
	return img[:, :, None]

def load_label(fname):
	img = np.loadtxt(fname)
	img /= 255.0
	return img[:, :, None]


imgnames = sorted(glob.glob('data/origs/*np'))
labelnames = sorted(glob.glob('data/labels/*np'))


assert len(imgnames) == len(labelnames)

X = [load_img(imname) for imname in imgnames]
Y = [load_label(imname) for imname in labelnames]

model = UNet()

imggen = ImageDataGenerator(
	rotation_range=20,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True
)

img_augmentation = imggen.random_transform


train_generator = generator(X, Y, batch_size=1, random=True, augment=img_augmentation)

callbacks = [
 	EarlyStopping(patience=50, monitor='loss'),
	ModelCheckpoint('unet.{epoch:03d}.{loss:.5f}.h5', monitor='loss', save_best_only=True),
]

model.fit_generator(
    train_generator,
    len(X),
    epochs=1000,
    verbose=1,
    callbacks=callbacks
)
