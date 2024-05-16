#!/usr/bin/env bash

WORK_DIR=${1:-/home/yibo/yolov8_train}
TEST_DIR=${2:-train_val_data}
DEFAULT_LABEL=${3:-xxxxxx} # 若不需要过滤则给个默认不存在的
FILTER_LABELS=${@:4}

cd $(dirname $0)/.. # open dir of the bash file

PYTHONPATH=$(pwd):$PYTHONPATH \
python3 tools/data_proc/filter_and_standard_label.py \
    --json-dir $WORK_DIR/$TEST_DIR \
    --filter-postfix-list MH-S1 MH-S2 KBJ \
    --filter-label-list $DEFAULT_LABEL $FILTER_LABELS
