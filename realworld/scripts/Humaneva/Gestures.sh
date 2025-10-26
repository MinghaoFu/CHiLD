#!/bin/bash

python -u main.py \
    --kld_weight 1e-7 \
    --dropout 0 \
    --learning_rate 0.0006 \
    --batch_size 64 \
    --is_training 1 \
    --root_path ./dataset/humaneva/ \
    --data_path Gestures.npy \
    --model_id Humaneva_Gestures \
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
    --layer 15 45 \
    --layer_nums 2