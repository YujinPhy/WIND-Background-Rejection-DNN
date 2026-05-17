#!/bin/bash

ES_H5="/workspace/Background-Rejection-for-Water-Cherenkov-Detector/raw_data/WIND_ES.h5"
N16_H5="/workspace/Background-Rejection-for-Water-Cherenkov-Detector/raw_data/WIND_16N.h5"

LOG_DIR="/home/Background-Rejection-for-Water-Cherenkov-Detector/logs"
LOG_NAME="HitMapCNN"

python train.py \
    --es-path "$ES_H5" \
    --n16-path "$N16_H5" \
    --log-path "$LOG_DIR" \
    --log-name "$LOG_NAME" \
    --seed 42 \
    --test-ratio 0.2 \
    --val-ratio 0.2 \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --shuffle \
    --num-workers 16 \
    --gpu \

