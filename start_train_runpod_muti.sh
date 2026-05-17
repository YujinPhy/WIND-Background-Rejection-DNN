#!/bin/bash

H5_PATH="/workspace/Background-Rejection-for-Water-Cherenkov-Detector/raw_data"

LOG_DIR="/home/Background-Rejection-for-Water-Cherenkov-Detector/logs"
# LOG_NAME="test"

python train.py \
    --es-path "$H5_PATH/WIND_ES_with_r_correction.h5" \
    --n16-path "$H5_PATH/WIND_16N_with_r_correction.h5" \
    --log-path "$LOG_DIR" \
    --log-name "r_correction" \
    --seed 42 \
    --test-ratio 0.2 \
    --val-ratio 0.2 \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --shuffle \
    --num-workers 16 \
    --gpu 

python train.py \
    --es-path "$H5_PATH/WIND_ES_with_z_correction.h5" \
    --n16-path "$H5_PATH/WIND_16N_with_z_correction.h5" \
    --log-path "$LOG_DIR" \
    --log-name "z_correction" \
    --seed 42 \
    --test-ratio 0.2 \
    --val-ratio 0.2 \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --shuffle \
    --num-workers 16 \
    --gpu 

python train.py \
    --es-path "$H5_PATH/WIND_ES_with_rz_corrections.h5" \
    --n16-path "$H5_PATH/WIND_16N_with_rz_corrections.h5" \
    --log-path "$LOG_DIR" \
    --log-name "rz_corrections" \
    --seed 42 \
    --test-ratio 0.2 \
    --val-ratio 0.2 \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --shuffle \
    --num-workers 16 \
    --gpu 