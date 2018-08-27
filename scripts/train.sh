#!/bin/bash
if [[ $VIRTUAL_ENV == "" ]]
then
    source ~/.virtualenvs/donkeycar/bin/activate
fi

MODEL="$HOME/models/mypilot"
TUB="$HOME/data/s5"
BATCH_SIZE="128"
EPOCHS="200"

python ~/custom/train.py --model $MODEL --tub $TUB --batch_size $BATCH_SIZE --epochs $EPOCHS
