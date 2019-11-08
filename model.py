import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Activation
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import load_model

from utils import psnr
from utils import load_test


class SRCNN:
    def __init__(self, image_size, c_dim, is_training, learning_rate=1e-4, batch_size=128, epochs=1500, name=''):
        self.image_size = image_size
        self.c_dim = c_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.is_training = is_training
        self.name = name
        '''if self.is_training:
            self.model = self.build_model()
        else:
            self.model = self.load()'''

    def build_model(self, name='srcnn'):
        self.name = name
        model = Sequential()
        model.add(Conv2D(64, 9, padding='same', input_shape=(
            self.image_size, self.image_size, self.c_dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.c_dim, 5, padding='same'))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error', metrics=['accuracy', psnr])
        self.model = model
        return model

    def build_model_filter_128_64(self, name):
        self.name = name
        model = Sequential()
        model.add(Conv2D(128, 9, padding='same', input_shape=(
            self.image_size, self.image_size, self.c_dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, 1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.c_dim, 5, padding='same'))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error', metrics=['accuracy', psnr])
        self.model = model
        return model

    def build_model_filter_32_16(self, name):
        self.name = name
        model = Sequential()
        model.add(Conv2D(32, 9, padding='same', input_shape=(
            self.image_size, self.image_size, self.c_dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(16, 1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.c_dim, 5, padding='same'))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error', metrics=['accuracy', psnr])
        self.model = model
        return model

    def build_model_9115(self, name):
        self.name = name
        model = Sequential()
        model.add(Conv2D(64, 9, padding='same', input_shape=(
            self.image_size, self.image_size, self.c_dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.c_dim, 5, padding='same'))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error', metrics=['accuracy', psnr])
        self.model = model
        return model

    def build_model_935(self, name):
        self.name = name
        model = Sequential()
        model.add(Conv2D(64, 9, padding='same', input_shape=(
            self.image_size, self.image_size, self.c_dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.c_dim, 5, padding='same'))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error', metrics=['accuracy', psnr])
        self.model = model
        return model

    def build_model_9315(self, name):
        self.name = name
        model = Sequential()
        model.add(Conv2D(64, 9, padding='same', input_shape=(
            self.image_size, self.image_size, self.c_dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.c_dim, 5, padding='same'))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error', metrics=['accuracy', psnr])
        self.model = model
        return model

    def build_model_955(self, name):
        self.name = name
        model = Sequential()
        model.add(Conv2D(64, 9, padding='same', input_shape=(
            self.image_size, self.image_size, self.c_dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 5, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.c_dim, 5, padding='same'))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error', metrics=['accuracy', psnr])
        self.model = model
        return model

    def build_model_9515(self, name):
        self.name = name
        model = Sequential()
        model.add(Conv2D(64, 9, padding='same', input_shape=(
            self.image_size, self.image_size, self.c_dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 5, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.c_dim, 5, padding='same'))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error', metrics=['accuracy', psnr])
        self.model = model
        return model

    def build_model_9115_16(self, name):
        self.name = name
        model = Sequential()
        model.add(Conv2D(64, 9, padding='same', input_shape=(
            self.image_size, self.image_size, self.c_dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(16, 1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.c_dim, 5, padding='same'))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error', metrics=['accuracy', psnr])
        self.model = model
        return model

    def build_model_91115(self, name):
        self.name = name
        model = Sequential()
        model.add(Conv2D(64, 9, padding='same', input_shape=(
            self.image_size, self.image_size, self.c_dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(16, 1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.c_dim, 5, padding='same'))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error', metrics=['accuracy', psnr])
        self.model = model
        return model

    def build_model_9335(self, name):
        self.name = name
        model = Sequential()
        model.add(Conv2D(64, 9, padding='same', input_shape=(
            self.image_size, self.image_size, self.c_dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.c_dim, 5, padding='same'))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error', metrics=['accuracy', psnr])
        self.model = model
        return model

    def build_model_9333(self, name):
        self.name = name
        model = Sequential()
        model.add(Conv2D(64, 9, padding='same', input_shape=(
            self.image_size, self.image_size, self.c_dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.c_dim, 3, padding='same'))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error', metrics=['accuracy', psnr])
        self.model = model
        return model

    def get_model_fns(self):
        return {
            'srcnn': self.build_model,
            'srcnn_filter_128_64': self.build_model_filter_128_64,
            'srcnn_filter_32_16': self.build_model_filter_32_16,
            'srcnn_9115': self.build_model_9115,
            'srcnn_935': self.build_model_935,
            'srcnn_9315': self.build_model_9315,
            'srcnn_955': self.build_model_955,
            'srcnn_9515': self.build_model_9515,
            'srcnn_9115_16': self.build_model_9115_16,
            'srcnn_91115': self.build_model_91115,
            'srcnn_9335': self.build_model_9335,
            'srcnn_9333': self.build_model_9333,
        }

    def save(self):
        name = self.name + '.h5'
        path = os.path.join('.\\model\\', name)
        self.model.save(path)
        return True

    def train(self, X_train, Y_train):
        # _, X_test, Y_test = load_test(scale=3)
        # X_test = [img.reshape(1, img.shape[0], img.shape[1], 1)
        #           for img in X_test]
        # validation_data = (X_test[0], Y_test[0])
        history = self.model.fit(X_train, Y_train, batch_size=self.batch_size,
                                 epochs=self.epochs, verbose=1, validation_split=0.1)  # validation_data=validation_data)
        if self.is_training:
            self.save()
        return history

    def process(self, img):
        predicted = self.model.predict(img)
        return predicted

    def load(self, name='srcnn.h5'):
        model = load_model(name)
        return model
