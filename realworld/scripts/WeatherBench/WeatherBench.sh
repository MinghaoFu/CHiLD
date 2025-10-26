#!/bin/bash

python -u main.py \
    --kld_weight 1e-7 \
    --dropout 0 \
    --learning_rate 0.0001 \
    --batch_size 64 \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path WeatherBench.csv \
    --model_id WeatherBench \
    --model CHiLD \
    --data WeatherBench \
    --features M \
    --seq_len 24 \
    --enc_in 104 \
    --dec_in 104 \
    --c_out 104 \
    --des "Exp" \
    --itr 1 \
    --train_epochs 1000 \
    --layer 14 90