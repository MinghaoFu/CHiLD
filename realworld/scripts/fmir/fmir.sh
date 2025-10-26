#!/bin/bash

python -u main.py \
    --kld_weight 1e-7 \
    --dropout 0 \
    --learning_rate 0.0001 \
    --batch_size 64 \
    --is_training 1 \
    --root_path ./dataset/fMRI/ \
    --data_path sim4.mat \
    --model_id fmri \
    --model CHiLD \
    --data fmri \
    --features M \
    --seq_len 24 \
    --enc_in 50 \
    --dec_in 50 \
    --c_out 50 \
    --des "Exp" \
    --itr 1 \
    --train_epochs 1000 \
    --layer 25 50 \
    --n_concat 3 \
    --layer_nums 2