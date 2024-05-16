#!/usr/bin/env bash

WORK_DIR=${1:-/home/yibo/yolov8_train}
ORIGIN_DIR=${2:-origin_data}
OUTPUT_DIR=${3:-train_val_data}
TRAIN_PERCENTAGE=${4:-0.8}
MIN_NUM_PER_CLASS=${5:-20}

cd $(dirname $0)/.. # open dir of the bash file
chmod 777 $WORK_DIR
echo "test:$(pwd):$PYTHONPATH"
PYTHONPATH=$(pwd):$PYTHONPATH \
python tools/data_proc/train_val_split_labelme.py \
    --input-dir $WORK_DIR/$ORIGIN_DIR \
    --output-dir $WORK_DIR/$OUTPUT_DIR \
    --train-percent $TRAIN_PERCENTAGE \
    --min-val-num $MIN_NUM_PER_CLASS
