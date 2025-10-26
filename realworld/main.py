import argparse
import torch
import torch.multiprocessing
import random
import os
import numpy as np
import time
from experiments.exp_CHiLD import Exp

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='iTransformer')

    # CHiLD
    parser.add_argument('--patch_size', type=int, default=5)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--c_type', type=str, default='type1')
    parser.add_argument('--layer', type=int, default=[], nargs='+', required=True, help='')
    parser.add_argument('--kld_weight', type=float, default=1e-7, help='num of encoder layers')
    parser.add_argument('--lags', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--n_concat', type=int, default=3, required=False)
    parser.add_argument('--layer_nums', type=int, default=3)
    parser.add_argument('--is_norm', action='store_false')
    parser.add_argument('--is_ln', action='store_false')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='CHiLD', help='model name')
    parser.add_argument('--test_epoch', type=int, default=100, required=False)
    parser.add_argument('--output', type=str, default='results', required=False)
    parser.add_argument('--filename', type=str, default=None, help='output filename')
    parser.add_argument('--seed', type=int, default=2024, help='seed')
    parser.add_argument('--metric', type=str, default='mae')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/human/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='WalkDog_all.npy', help='data csv/npy file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7,
                        help='output size')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


    


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('Args in experiment:')
    torch.multiprocessing.set_sharing_strategy('file_system')
    if args.is_training == 1:
        for ii in range(args.itr):
            print(args)
            # setting record of experiments
            setting = f'{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}' \
                      f'_lr{args.learning_rate}_bs{args.batch_size}_seed{args.seed}' \
                      f'_layer{args.layer}'
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = f'{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}' \
                  f'_lr{args.learning_rate}_bs{args.batch_size}_seed{args.seed}' \
                  f'_layer{args.layer}'
        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
