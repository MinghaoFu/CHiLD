#!/bin/bash

python -u main.py \
    --kld_weight 1e-7 \
    --dropout 0 \
    --learning_rate 0.001 \
    --batch_size 128 \
    --is_training 1 \
    --root_path ./dataset/MuJoco/ \
    --data_path MuJoco \
    --model_id MuJoco \
    --model CHiLD \
    --data MuJoco \
    --features M \
    --seq_len 24 \
    --enc_in 14 \
    --dec_in 14 \
    --c_out 14 \
    --des "Exp" \
    --itr 1 \
    --train_epochs 500 \
    --layer 7 14 \
    --n_concat 1 \
    --layer_nums 1