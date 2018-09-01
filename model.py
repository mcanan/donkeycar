import time
import numpy as np
from donkeycar.parts.keras import KerasPilot
from PIL import Image
try:
    from tensorflow.python.keras.models import Model
    from tensorflow.python.keras.layers import (Input, Dropout, Flatten,
                                                Dense, Convolution2D,
                                                BatchNormalization,
                                                MaxPooling2D)
    from tensorflow.python.keras.optimizers import SGD, Adam
except ImportError:
    from keras.models import Model
    from keras.layers import (Input, Dropout, Flatten, Dense, Convolution2D,
                              BatchNormalization, MaxPooling2D)
    from keras.optimizers import SGD, Adam

ANGLE_CATEGORIES = 15
THROTTLE_CATEGORIES = 5


class MyModel(KerasPilot):
    def __init__(self, min_throttle=0., max_throttle=.5,
                 joystick_max_throttle=.5,
                 verbose=0, *args, **kwargs):
        super(MyModel, self).__init__(*args, **kwargs)
        self.model = my_model_all_categorical(input_shape=(75, 160, 3))
        self.categorical_angle = True
        self.categorical_throttle = True
        self.angle_postprocessing = PostprocessAngle(verbose)
        self.throttle_postprocessing = PostprocessThrottle(
                                       min_throttle=min_throttle,
                                       max_throttle=max_throttle,
                                       joystick_max_throttle=joystick_max_throttle,
                                       verbose=verbose)
        self.last_image_ts = None
        self.diff_ts = []
        self.verbose = verbose

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle, throttle = self.model.predict(img_arr)
        a = linear_unbin(angle[0], (-1, 1), ANGLE_CATEGORIES) \
            if self.categorical_angle else angle[0][0]
        t = linear_unbin(throttle[0], (0, 1), THROTTLE_CATEGORIES) \
            if self.categorical_throttle else throttle[0][0]
        a = self.angle_postprocessing.run(a)
        t = self.throttle_postprocessing.run(t)
        if self.verbose:
            current_ts = int(round(time.time() * 1000))
            if self.last_image_ts is not None:
                self.diff_ts.append(current_ts - self.last_image_ts)
                if len(self.diff_ts) == 100:
                    print("Hz:", int(1000./np.array(self.diff_ts).mean()))
                    self.diff_ts = []
            self.last_image_ts = current_ts
        return a, t


class PostprocessAngle():
    def __init__(self, verbose):
        self.verbose = verbose

    def run(self, angle):
        return angle


class PostprocessThrottle():
    def __init__(self, min_throttle, max_throttle, joystick_max_throttle,
                 verbose):
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.joystick_max_throttle = joystick_max_throttle
        self.verbose = verbose
        self.last_t = 0

    def run(self, throttle):
        t = self.min_throttle + (self.max_throttle - self.min_throttle) * \
            (throttle / self.joystick_max_throttle)
        if self.verbose and t != self.last_t:
            print("Original thr:", throttle, "Postprocessed thr:", t)
            self.last_t = t
        return t


class PreprocessImage():
    def run(self, img):
        img = Image.fromarray(img)
        img = img.crop((0, 45, 160, 120))
        return np.array(img)/127.5-1.


def my_model_all_categorical(input_shape):
    img_in = Input(shape=input_shape, name='img_in')
    x = Convolution2D(32, (3, 3), padding='same', strides=(2, 2),
                      activation='relu')(img_in)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(x)
    x = Convolution2D(64, (3, 3), padding='same', strides=(2, 2),
                      activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Convolution2D(128, (3, 3), padding='same', strides=(1, 1),
                      activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)

    x = Flatten(name='flattened')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(50, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(50, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    angle_out = Dense(15, activation='softmax', name='angle_out')(x)
    throttle_out = Dense(5, activation='softmax', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    optimizer = Adam()
    model.compile(optimizer=optimizer,
                  loss={'angle_out': 'categorical_crossentropy',
                        'throttle_out': 'categorical_crossentropy'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': 0.5})

    return model


def linear_bin(a, rng, size):
    r = float(rng[1] - rng[0])
    b = round((a - rng[0]) / (r / (size-1)))
    arr = np.zeros(size)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr, rng, size):
    r = float(rng[1] - rng[0])
    b = np.argmax(arr)
    a = rng[0] + b * (r / (size-1))
    return a
