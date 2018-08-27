#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it.

Usage:
    train.py [--tub=<tub1,tub2,..tubn>] (--model=<model>)
             [--batch_size=<batch_size>] [--epochs=<epochs>]

Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. Comma separated.
                     Use quotes to use wildcards. ie "~/tubs/*"
"""
import glob
import json
import os
import numpy as np
import re
from docopt import docopt
from PIL import Image, ImageEnhance, ImageFilter
from tensorflow.python.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                               CSVLogger, ReduceLROnPlateau)

import donkeycar as dk
from model import (MyModel, PreprocessImage, linear_bin, linear_unbin,
                   ANGLE_CATEGORIES, THROTTLE_CATEGORIES)
from utils import OutputCallback

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

def tubs_to_arrays(paths, seed=None):
    images, y = [], []
    img_ok, json_error = 0, 0

    tub_paths = paths.split(',')
    tub_paths = sorted(tub_paths)

    for path in tub_paths:
        files = glob.glob(os.path.join(path, 'record*.json'))
        files = sorted(files, key=lambda x: int(re.findall(r'\d+',
                       os.path.basename(x))[0]))
        for filename in files:
            with open(filename, encoding='utf-8') as data_file:
                try:
                    data = json.loads(data_file.read())
                except:
                    print("Error loading json", filename)
                    json_error += 1
                    continue
                img_file = os.path.join(path, data['cam/image_array'])
                if not os.path.isfile(img_file):
                    continue

            img_ok += 1
            angle = data['user/angle']
            throttle = data['user/throttle']
            y.append([angle, throttle])
            img = Image.open(os.path.join(path, data['cam/image_array']))
            images.append(np.array(img))

    X = np.array(images)
    y = np.array(y)

    if seed is not None:
        np.random.seed(seed)
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

    print('Images ok: %d, Json error: %d' % (img_ok, json_error))
    print('Mean: %s' % X.mean(axis=(0, 1, 2), dtype=np.int))

    return X, y


def generator(X, y, batch_size, train, categorical_angle, categorical_throttle):
    idx = np.arange(X.shape[0])
    pi = PreprocessImage()
    while 1:
        if train:
            np.random.shuffle(idx)
        cnt = np.zeros(ANGLE_CATEGORIES)
        images, angles, throttles = [], [], []
        for i in idx:
            angle = y[i, 0]
            flip = False

            if train and np.random.rand() > .5:
                angle = -1 * angle
                flip = True

            angle_bin = linear_bin(angle, (-1, 1), ANGLE_CATEGORIES)
            if train and cnt[angle_bin.argmax()] > batch_size//ANGLE_CATEGORIES:
                continue
            cnt[angle_bin.argmax()] += 1

            image = Image.fromarray(X[i])

            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            if train:
                if np.random.rand() > .5:
                    image = ImageEnhance.Brightness(image).enhance(
                            np.random.uniform(.5, 2))
                if np.random.rand() > .5:
                    image = ImageEnhance.Contrast(image).enhance(
                            np.random.uniform(.5, 2))
                if np.random.rand() > .75:
                    image = image.filter(ImageFilter.BLUR)

            image = pi.run(np.array(image))

            if categorical_angle:
                angle = linear_bin(angle, (-1, 1), ANGLE_CATEGORIES)

            throttle = y[i, 1]
            if categorical_throttle:
                throttle = linear_bin(throttle, (0, 1), THROTTLE_CATEGORIES)

            images.extend([image])
            angles.extend([angle])
            throttles.extend([throttle])
            if len(images) == batch_size:
                break

        X_batch = np.array(images, dtype=np.float32)
        y_batch_1 = np.array(angles, dtype=np.float32)
        y_batch_2 = np.array(throttles, dtype=np.float32)

        yield X_batch, [y_batch_1, y_batch_2]


def train(tub_names, model_path, batch_size, epochs):
    model_path = os.path.expanduser(model_path)
    m = MyModel()
    model = m.model
    model.summary()
    X, y = tubs_to_arrays(tub_names, seed=10)

    total_records = len(X)
    total_train = int(total_records * .8)
    total_val = total_records - total_train
    steps_per_epoch = ((total_train // batch_size) + 1)*2
    validation_steps = (total_val // batch_size) + 1

    print('Train images: %d, Validation images: %d' % (total_train, total_val))
    print('Batch size:', batch_size)
    print('Epochs:', epochs)
    print('Training steps:', steps_per_epoch)
    print('Validation steps:', validation_steps)

    input("Press Enter to continue...")

    train_gen = generator(X[:total_train], y[:total_train], batch_size,
                          train=True, categorical_angle=m.categorical_angle,
                          categorical_throttle=m.categorical_throttle)
    val_gen = generator(X[total_train:], y[total_train:], batch_size,
                        train=False, categorical_angle=m.categorical_angle,
                        categorical_throttle=m.categorical_throttle)

    save_best = ModelCheckpoint(model_path,
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True,
                                mode='min')

    callbacks = [save_best, CSVLogger("logs/train.log"), OutputCallback()]

    hist = model.fit_generator(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=0,
        validation_data=val_gen,
        callbacks=callbacks,
        validation_steps=validation_steps,
        workers=4,
        use_multiprocessing=True)
    return hist


if __name__ == '__main__':
    args = docopt(__doc__)

    tub = args['--tub']
    model_path = args['--model']
    batch_size = int(args['--batch_size'])
    epochs = int(args['--epochs'])

    train(tub, model_path, batch_size, epochs)
