#!/bin/bash

python -u main.py \
    --kld_weight 1e-7 \
    --dropout 0 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id ETTh1 \
    --model CHiLD \
    --data ETTh1 \
    --features M \
    --seq_len 24 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des "Exp" \
    --itr 1 \
    --train_epochs 800 \
    --layer 3 7 \
    --n_concat 1