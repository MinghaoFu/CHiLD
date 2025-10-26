#!/bin/bash

python -u main.py \
    --kld_weight 1e-7 \
    --dropout 0 \
    --learning_rate 0.0006 \
    --batch_size 64 \
    --is_training 1 \
    --root_path ./dataset/human/ \
    --data_path Purchases_all.npy \
    --model_id Purchases \
    --model CHiLD \
    --data Human \
    --features M \
    --seq_len 24 \
    --layer 25 51 \
    --enc_in 51 \
    --dec_in 51 \
    --c_out 51 \
    --des "Exp" \
    --itr 1 \
    --train_epochs 1500 \
    --n_concat 3