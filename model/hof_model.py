#!/usr/bin/env conda run -n AIMethods python
import pandas as pd
import numpy as np
import mnist
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# The first time you run this might be a bit slow, since the
# mnist package has to download and cache the data.
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

print(train_images.shape) # (60000, 28, 28)
tf.print("Train images: ", train_images) # (60000,)
print(train_labels.shape) # (60000,)
tf.print("Train labels: ", train_labels) # (60000,)


model = Sequential([
    Dense(10, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

test_cat = to_categorical(test_labels)
print("testing to_categorical", test_cat)


# model.fit(
#     train_data,
