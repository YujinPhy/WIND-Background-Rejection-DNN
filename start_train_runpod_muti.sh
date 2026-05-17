#!/bin/bash

H5_PATH="/workspace/Background-Rejection-for-Water-Cherenkov-Detector/raw_data"

LOG_DIR="/workspace/Background-Rejection-for-Water-Cherenkov-Detector/logs"

IN_CH=2
IMAGE_H=91
IMAGE_W=142


python train.py \
    --es-path "$H5_PATH/WIND_ES_with_r_correction.h5" \
    --n16-path "$H5_PATH/WIND_16N_with_r_correction.h5" \
    --in-ch $IN_CH \
    --image-h $IMAGE_H \
    --image-w $IMAGE_W \
    --num-workers 16 \
    --gpu \
    --log-path "$LOG_DIR" \
    --log-name "r_correction" \
    --model-name "HitMapCNN" \
    --seed 42 \
    --test-ratio 0.2 \
    --val-ratio 0.2 \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --shuffle \
    --target-bkg-residual 0.03

python train.py \
    --es-path "$H5_PATH/WIND_ES_with_z_correction.h5" \
    --n16-path "$H5_PATH/WIND_16N_with_z_correction.h5" \
    --in-ch $IN_CH \
    --image-h $IMAGE_H \
    --image-w $IMAGE_W \
    --num-workers 16 \
    --gpu \
    --log-path "$LOG_DIR" \
    --log-name "z_correction" \
    --model-name "HitMapCNN" \
    --seed 42 \
    --test-ratio 0.2 \
    --val-ratio 0.2 \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --shuffle \
    --target-bkg-residual 0.03

python train.py \
    --es-path "$H5_PATH/WIND_ES_with_rz_corrections.h5" \
    --n16-path "$H5_PATH/WIND_16N_with_rz_corrections.h5" \
    --in-ch $IN_CH \
    --image-h $IMAGE_H \
    --image-w $IMAGE_W \
    --num-workers 16 \
    --gpu \
    --log-path "$LOG_DIR" \
    --log-name "rz_corrections" \
    --model-name "HitMapCNN" \
    --seed 42 \
    --test-ratio 0.2 \
    --val-ratio 0.2 \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --shuffle \
    --target-bkg-residual 0.03