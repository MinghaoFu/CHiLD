#!/bin/bash

python -u main.py \
    --kld_weight 1e-7 \
    --dropout 0 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path stock_data.csv \
    --model_id stocks \
    --model CHiLD \
    --data custom \
    --features M \
    --seq_len 24 \
    --enc_in 6 \
    --dec_in 6 \
    --c_out 6 \
    --des "Exp" \
    --itr 1 \
    --train_epochs 500 \
    --layer 3 6 \
    --n_concat 3