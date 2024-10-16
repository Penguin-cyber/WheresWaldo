# Libraries
import io
import sys
import time
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

# The actual machine learning libraries
import tensorflow as tf
from keras import layers, models, initializers
import keras
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def genImage():
    # Image setup
    image_dir = os.getcwd() + "/Images"

    ship_dir = image_dir + "/ship.jpg"
    waldo_dir = image_dir + "/waldo.png"

    # gathers the images
    ship_im = Image.open(ship_dir)
    waldo_im = Image.open(waldo_dir)

    # resizes images
    ship_im = ship_im.resize((1024, 768))
    waldo_im = waldo_im.resize((70, 120))

    # random time!!
    col = np.random.randint(0, 950)
    row = np.random.randint(0, 700)

    ship_im.paste(waldo_im, (col, row), mask=waldo_im)
    return np.array(ship_im).astype("uint8"), (col, row)


def genData(batch_size=24):

    while True:
        # generates the empty arrays for the data
        x_batch = np.zeros((batch_size, 768, 1024, 3))
        y_batch = np.zeros((batch_size, 2))

        for i in range(batch_size):
            sample_im, ans = genImage()

            x_batch[i] = sample_im/255  # normalizes to [0, 1]
            y_batch[i, 0] = ans[0]
            y_batch[i, 1] = ans[1]

        yield {"images": x_batch}, {"boxes": y_batch}


# Generating the model
model = keras.models.Sequential()

# input image is RGB and 1024 x 768
model.add(keras.Input(shape=(768, 1024, 3), name="images"))

# Convolution + ReLU + Batch Normalization + Max Pooling
model.add(layers.Conv2D(16, 3, strides=1,
          padding="same", activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, strides=2))

model.add(layers.Conv2D(32, 3, strides=1,
          padding="same", activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, strides=2))

model.add(layers.Conv2D(64, 3, strides=1,
          padding="same", activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, strides=2))

model.add(layers.Conv2D(128, 3, strides=1,
          padding="same", activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, strides=2))

model.add(layers.Conv2D(256, 3, strides=1,
          padding="same", activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, strides=2))

model.add(layers.Conv2D(512, 3, strides=1,
          padding="same", activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, strides=2))

model.add(layers.Conv2D(512, 3, strides=1,
          padding="same", activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, strides=2))

# Flatten + Fully Connected Layers
model.add(layers.Flatten())  # converting the tensor into a vector

model.add(layers.Dense(512, activation="relu"))

model.add(layers.Dense(64, activation="relu"))

model.add(layers.Dense(2, name="boxes"))  # outputs 2 values

# Using ADAM to optimize and Mean Squared Loss
model.compile(optimizer=keras.optimizers.AdamW(),
              loss=keras.losses.mean_squared_error)

model.summary()

# START TRAINING

# start timer
tick = time.time()

history = model.fit(genData(), steps_per_epoch=150,
                    epochs=15)  # fits the model

# stop timer
tock = time.time()

print('Took {} minutes to run finish training 10 epoch(s)'.format(
    np.round((tock - tick)/60, 2)))

model.save("waldo_model.keras")
