#!/usr/bin/env sh
set -e

echo "convert:............."
DATASET_DIR=./data/sim_sub_15w/
OUTPUT_DIR=./data/output/

python convertData.py \
    --dataset_dir=${DATASET_DIR} \
    --name=ocr \
    --output_dir=${OUTPUT_DIR} \
    --str_size=10 \
    --split=True