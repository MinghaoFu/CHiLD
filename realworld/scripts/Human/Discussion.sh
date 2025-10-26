#!/bin/bash

python -u main.py \
    --kld_weight 1e-7 \
    --dropout 0 \
    --learning_rate 0.0001 \
    --batch_size 64 \
    --is_training 1 \
    --root_path ./dataset/human/ \
    --data_path Discussion_all.npy \
    --model_id Discussion \
    --model CHiLD \
    --data Human \
    --features M \
    --seq_len 24 \
    --layer 17 51 \
    --enc_in 51 \
    --dec_in 51 \
    --c_out 51 \
    --des "Exp" \
    --itr 1 \
    --train_epochs 1000 \
    --n_concat 3 \
    --layer_nums 2