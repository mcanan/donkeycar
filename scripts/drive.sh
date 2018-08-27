#!/bin/bash
CONFIG="$HOME/custom/m1/config.py"
MODEL="$HOME/models/mypilot"
MIN_THROTTLE="0.35"
MAX_THROTTLE="0.4"
VERBOSE="0"

if [ ! -c /dev/input/js0 ]; then
   echo "The PS3 controller is not connected."
   exit
fi

python ~/custom/drive.py --config=$CONFIG --model=$MODEL --min-throttle=$MIN_THROTTLE \
    --max-throttle=$MAX_THROTTLE --verbose=$VERBOSE
