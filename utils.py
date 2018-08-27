#!/usr/bin/env python3
import glob
import json
import os
import re
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageDraw
from matplotlib import pyplot as plt
from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.callbacks import Callback
from random import shuffle
from time import time

import donkeycar as dk
from model import PreprocessImage, MyModel


def show_predictions(model_path, tub_paths, start=0, end=100, index=0):
    images, y, predictions = [], [], []
    img_ok = 0

    model = MyModel(min_throttle=0., max_throttle=1.)
    model.load(model_path)
    pi = PreprocessImage()

    for path in tub_paths:
        files = glob.glob(os.path.join(path, 'record*.json'))
        for filename in files:
            with open(filename, encoding='utf-8') as data_file:
                data = json.loads(data_file.read())
                if os.path.isfile(os.path.join(path, data['cam/image_array'])):
                    img_ok += 1
                    y.append([data['user/angle'], data['user/throttle']])
                    img = Image.open(os.path.join(path,
                                     data['cam/image_array']))
                    predictions.append(model.run(pi.run(np.array(img))))
                    img = np.array(img)
                    images.append(img)

    images = np.array(images)
    y = np.array(y)
    predictions = np.array(predictions)

    fig, ax = plt.subplots()
    plt.plot(y[start:end, index])
    plt.plot(predictions[start:end, index])
    plt.show()


def show_histogram(tub_paths, start=0, end=100000, index=0, ax=None):
    y = []
    img_ok = 0

    for path in tub_paths:
        files = glob.glob(os.path.join(path, 'record*.json'))
        for filename in files:
            with open(filename, encoding='utf-8') as data_file:
                data = json.loads(data_file.read())
                if os.path.isfile(os.path.join(path, data['cam/image_array'])):
                    img_ok += 1
                    y.append([data['user/angle'], data['user/throttle']])

    y = np.array(y)

    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(y[start:end, index])


def show_batch_histogram(tub_paths, batch_size, train=True, ax=None):
    y = []
    img_ok = 0

    for path in tub_paths:
        files = glob.glob(os.path.join(path, 'record*.json'))
        for filename in files:
            with open(filename, encoding='utf-8') as data_file:
                data = json.loads(data_file.read())
                if os.path.isfile(os.path.join(path, data['cam/image_array'])):
                    y.append([data['user/angle'], data['user/throttle']])
    shuffle(y)
    y = np.array(y)
    cnt = np.zeros(15)
    batch = []
    for i in range(len(y)):
        angle = y[i][0]
        if train and np.random.rand() > .5:
            angle = -1 * angle

        angle_bin = dk.util.data.linear_bin(angle)
        if train and cnt[angle_bin.argmax()] > batch_size//15:
            continue
        cnt[angle_bin.argmax()] += 1
        batch.append(y)
        if len(y) == batch_size:
            break

    if ax is None:
        fig, ax = plt.subplots()
    ax.bar(np.arange(len(cnt)), cnt)


def make_video(tub_path, video_filename='video.avi', model_path=None,
               preprocess_angle=None, index=None, min_throttle=0.,
               max_throttle=1.):
    files = glob.glob(os.path.join(tub_path, 'record*.json'))
    files = sorted(files, key=lambda x: int(re.findall(r'\d+',
                   os.path.basename(x))[0]))

    if model_path is not None:
        model = MyModel(min_throttle=min_throttle, max_throttle=max_throttle)
        model.load(model_path)

    pi = PreprocessImage()
    video = None
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    for filename in files:
        with open(filename, encoding='utf-8') as data_file:
            data = json.loads(data_file.read())
            if os.path.isfile(os.path.join(tub_path, data['cam/image_array'])):
                frame = cv2.imread(os.path.join(tub_path,
                                                data['cam/image_array']))
                throttle = data['user/throttle']
                angle = data['user/angle']
                xa = int(frame.shape[1] * ((angle+1)/2.))
                ya = int(frame.shape[0] * .95)
                xt = int(frame.shape[1] * .95)
                yt = int(frame.shape[0] - frame.shape[0] * throttle)
                if index is None or index == 0:
                    cv2.circle(frame, (xa, ya), 2, (255, 128, 0), -1)
                if index is None or index == 1:
                    cv2.circle(frame, (xt, yt), 2, (255, 128, 0), -1)
                if model_path is not None:
                    img = Image.open(os.path.join(tub_path,
                                                  data['cam/image_array']))
                    p_angle, p_throttle = model.run(pi.run(np.array(img)))
                    if preprocess_angle is not None:
                        p_angle = preprocess_angle(p_angle)
                    xa = int(frame.shape[1] * ((p_angle+1)/2.))
                    ya = int(frame.shape[0] * .9)
                    xt = int(frame.shape[1] * .9)
                    yt = int(frame.shape[0] - frame.shape[0] * p_throttle)
                    if index is None or index == 0:
                        cv2.circle(frame, (xa, ya), 2, (0, 128, 255), -1)
                    if index is None or index == 1:
                        cv2.circle(frame, (xt, yt), 2, (0, 128, 255), -1)
                if video is None:
                    h, w, ch = frame.shape
                    video = cv2.VideoWriter(video_filename,
                                            fourcc, 20., (w, h))
                video.write(frame)
    cv2.destroyAllWindows()
    video.release()


class OutputCallback(Callback):
    def __init__(self):
        self.seen = 0
        self.epoch = 0
        self.best_val_loss = 1e10
 
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime=time()

    def on_batch_end(self, epoch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size

    def on_epoch_end(self, epoch, logs={}):
        elapsed_ms = (time()-self.starttime)
        time_1000_imgs = 1000 * (elapsed_ms/self.seen)
        val_loss = logs.get('val_loss', 0)
        loss = logs.get('loss', 0)
        print("epoch:","%04d" % self.epoch,
              "images:", self.seen,
              "time:", "%.2f" % round(time_1000_imgs, 2), 
              "loss:", "%.3f" % round(loss, 3),
              "val_loss:", "%.3f" % round(val_loss, 3), "*" if val_loss < self.best_val_loss else "")
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        self.seen = 0
        self.epoch += 1
        
