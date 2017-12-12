# Code heavily borrowed from Keras blog post:
# - https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# - https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

# fix broken division; this script was written for Python 3,
# but we're running it on Python 2
from __future__ import division

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# dimensions of our images.
img_width = 32
img_height = 32

model_save_path = 'bottleneck_fc_full_model.h5'
train_data_dir = '../training/labeled-split-images/train'
validation_data_dir = '../training/labeled-split-images/validation'
nb_train_samples = 496
nb_validation_samples = 192
epochs = 10
batch_size = 16

# JJ TODO don't hardcode
nb_train_r = 129
nb_train_y = 62
nb_train_g = 305

# JJ TODO don't hardcode
nb_val_r = 58
nb_val_y = 17
nb_val_g = 117

def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=['0', '1', '2'],
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size, verbose=1)
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        classes=['0', '1', '2'],
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size, verbose=1)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)

def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    # JJ TODO don't hardcode this
    train_labels = np.array(([[1, 0, 0]] * nb_train_r) + ([[0, 1, 0]] * nb_train_y) + ([[0, 0, 1]] * nb_train_g))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    # JJ TODO don't hardcode this
    validation_labels = np.array(([[1, 0, 0]] * nb_val_r) + ([[0, 1, 0]] * nb_val_y) + ([[0, 0, 1]] * nb_val_g))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              verbose=2,
              validation_data=(validation_data, validation_labels))
    model.save(model_save_path)

save_bottleneck_features()
train_top_model()
