#!/bin/bash

python -u main.py \
    --kld_weight 1e-7 \
    --dropout 0 \
    --learning_rate 0.0006 \
    --batch_size 64 \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id weather \
    --model CHiLD \
    --data weather \
    --features M \
    --seq_len 24 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des "Exp" \
    --itr 1 \
    --train_epochs 1000 \
    --layer 7 14 \
    --layer_nums 2