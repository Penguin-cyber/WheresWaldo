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


def genBox(im, ans, guess=None):
    # full screens the image to be displayed
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    # conv im to array
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    draw.rectangle((ans[0]-10, ans[1]-10, ans[0] + 80,
                   ans[1] + 130), outline="green", width=5)

    if guess is not None:
        draw.rectangle((guess[0]-10, guess[1]-10, guess[0] + 80,
                        guess[1] + 130), outline="red", width=5)

    return im


def genTest(batch_size=16):
    # generates the empty arrays for the data
    x_batch = np.zeros((batch_size, 768, 1024, 3))
    y_batch = np.zeros((batch_size, 2))

    for i in range(batch_size):
        sample_im, ans = genImage()

        x_batch[i] = sample_im/255  # normalizes to [0, 1]
        y_batch[i, 0] = ans[0]
        y_batch[i, 1] = ans[1]

    return x_batch, y_batch


def testModel():
    images, answers = genTest()

    test_answer = model.predict(images, steps=3)

    for i in range(3):
        lbl_image = genBox((images[i]*255).astype("uint8"),
                           (answers[i]), (test_answer[i]))

        plt.imshow(lbl_image)
        plt.yticks([])
        plt.xticks([])
        plt.show()


# TESTING MODEL
model = keras.models.load_model("waldo_model.keras")

testModel()
