#!/bin/bash

python -u main.py \
    --kld_weight 1e-7 \
    --dropout 0 \
    --learning_rate 0.0003 \
    --batch_size 64 \
    --is_training 1 \
    --root_path ./dataset/humaneva/ \
    --data_path Box.npy \
    --model_id Humaneva_Box \
    --model CHiLD \
    --data Humaneva \
    --features M \
    --seq_len 24 \
    --enc_in 45 \
    --dec_in 45 \
    --c_out 45 \
    --des "Exp" \
    --itr 1 \
    --train_epochs 1000 \
    --layer 15 30 \
    --layer_nums 4