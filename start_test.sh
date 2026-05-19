#!/bin/bash

ES_H5="/home/yujin/projects/wind/WIND_bkg_rejection/raw_data/WIND_ES.h5"
N16_H5="/home/yujin/projects/wind/WIND_bkg_rejection/raw_data/WIND_16N.h5"

LOG_DIR="/home/yujin/projects/wind/WIND_bkg_rejection/logs"
LOG_NAME="HitMapCNN_default"

IN_CH=2
IMAGE_H=91
IMAGE_W=142

python test_evaluate.py \
    --es-path "$ES_H5" \
    --n16-path "$N16_H5" \
    --in-ch $IN_CH \
    --image-h $IMAGE_H \
    --image-w $IMAGE_W \
    --num-workers 16 \
    --log-path "$LOG_DIR" \
    --log-name "$LOG_NAME" \
    --model-name "HitMapCNN" \
    --seed 42 \
    --test-ratio 0.2 \
    --val-ratio 0.2 \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --shuffle \
    --target-bkg-residual 0.03
    # --gpu \

    