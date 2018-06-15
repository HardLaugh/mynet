#!/usr/bin/env sh
set -e

echo "train:............."
DATASET_DIR=../data/output/
FP=ocr_train_*.tfrecord
TC=../log/checkpoints/
SD=../log/summaries/
#CHECKPOINT_PATH=None

python train.py \
    --dataset_dir=${DATASET_DIR} \
    --file_pattern=${FP} \
    --train_checkpoints=${TC} \
    --summaries_dir=${SD} \
    --log_every_n_steps=60 \
    --number_of_steps=40000 \
    --batch_size=32 \
    --learning_rate=0.1