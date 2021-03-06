#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car.

Usage:
    drive.py --config=<config> [--model=<model>] [--js] [--chaos]
             [--min-throttle=<min>] [--max-throttle=<max>]
             [--verbose=<verbose>]

Options:
    -h --help        Show this screen.
    --config CONFIG  Config file to use.
    --model MODEL    Model to use in pilot drive.
    --js             Use physical joystick.
    --chaos          Add periodic random steering when manually driving
    --verbose V      Verbose mode
"""
import os
from docopt import docopt

import donkeycar as dk

from donkeycar.parts.camera import PiCamera
from donkeycar.parts.transform import Lambda
from donkeycar.parts.keras import KerasCategorical
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from donkeycar.parts.datastore import TubGroup, TubWriter
from donkeycar.parts.controller import LocalWebController, JoystickController

from model import PreprocessImage, MyModel


def drive(cfg, model_path=None, use_joystick=False, use_chaos=False,
          min_throttle=0, max_throttle=.5, verbose=0):
    """
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor
    flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a
    timely manner.
    Parts may have named outputs and inputs. The framework handles passing
    named outputs to parts requesting the same named input.
    """

    V = dk.vehicle.Vehicle()

    cam = PiCamera(resolution=cfg.CAMERA_RESOLUTION)
    V.add(cam, outputs=['cam/image_array'], threaded=True)

    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        ctr = JoystickController(max_throttle=cfg.JOYSTICK_MAX_THROTTLE,
                                 steering_scale=cfg.JOYSTICK_STEERING_SCALE,
                                 auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE)
    else:
        # This web controller will create a web server that is capable
        # of managing steering, throttle, and modes, and more.
        ctr = LocalWebController(use_chaos=use_chaos)

    V.add(ctr,
          inputs=['cam/image_array'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)

    # See if we should even run the pilot module.
    # This is only needed because the part run_condition only accepts boolean
    def pilot_condition(mode):
        if mode == 'user':
            return False
        else:
            return True

    pilot_condition_part = Lambda(pilot_condition)
    V.add(pilot_condition_part, inputs=['user/mode'],
          outputs=['run_pilot'])

    # Preprocess image if the mode is not user.
    pi = PreprocessImage()
    V.add(pi, inputs=['cam/image_array'],
          outputs=['cam/image_array_preprocessed'],
          run_condition='run_pilot')

    # Run the pilot if the mode is not user.
    kl = MyModel(min_throttle=min_throttle,
                 max_throttle=max_throttle,
                 joystick_max_throttle=cfg.JOYSTICK_MAX_THROTTLE,
                 verbose=verbose)
    if model_path:
        kl.load(model_path)

    V.add(kl, inputs=['cam/image_array_preprocessed'],
          outputs=['pilot/angle', 'pilot/throttle'],
          run_condition='run_pilot')

    # Choose what inputs should change the car.
    def drive_mode(mode,
                   user_angle, user_throttle,
                   pilot_angle, pilot_throttle):
        if mode == 'user':
            return user_angle, user_throttle

        elif mode == 'local_angle':
            return pilot_angle, user_throttle

        else:
            return pilot_angle, pilot_throttle

    drive_mode_part = Lambda(drive_mode)
    V.add(drive_mode_part,
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'],
          outputs=['angle', 'throttle'])

    steering_controller = PCA9685(cfg.STEERING_CHANNEL)
    steering = PWMSteering(controller=steering_controller,
                           left_pulse=cfg.STEERING_LEFT_PWM,
                           right_pulse=cfg.STEERING_RIGHT_PWM)

    throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL)
    throttle = PWMThrottle(controller=throttle_controller,
                           max_pulse=cfg.THROTTLE_FORWARD_PWM,
                           zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                           min_pulse=cfg.THROTTLE_REVERSE_PWM)

    V.add(steering, inputs=['angle'])
    V.add(throttle, inputs=['throttle'])

    # add tub to save data
    inputs = ['cam/image_array', 'user/angle', 'user/throttle', 'user/mode',
              'timestamp']
    types = ['image_array', 'float', 'float',  'str', 'str']

    # multiple tubs
    # th = TubHandler(path=cfg.DATA_PATH)
    # tub = th.new_tub_writer(inputs=inputs, types=types)

    # single tub
    print("Tub:", cfg.TUB_PATH)
    tub = TubWriter(path=cfg.TUB_PATH, inputs=inputs, types=types)
    V.add(tub, inputs=inputs, run_condition='recording')

    # run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config(args['--config'])
    model_path = args['--model']
    use_joystick = args['--js']
    use_chaos = args['--chaos']
    min_throttle = float(args['--min-throttle'])
    max_throttle = float(args['--max-throttle'])
    verbose = int(args['--verbose'])

    drive(cfg, model_path=model_path, use_joystick=use_joystick,
          use_chaos=use_chaos, min_throttle=min_throttle,
          max_throttle=max_throttle, verbose=verbose)
