"""
最小复现入口。仅支持 classification 任务 + UEA 数据 + M2SFormerATPSelfRegulationSCP2 模型。

示例（测试，加载已有 checkpoint）:
    HP__DROPOUT=0.2 HP__HIDDEN_DIM=256 HP__SEQ_LEN=5000 HP__D_MODEL=512 \
    python run.py --task_name classification --is_training 0 \
        --root_path ./dataset/UWaveGestureLibrary/ --model_id UWaveGestureLibrary \
        --model M2SFormerATPSelfRegulationSCP2 --data UEA --e_layers 3 \
        --batch_size 256 --d_model 128 --d_ff 256 --top_k 3 --des Exp --itr 1 \
        --learning_rate 0.001 --train_epochs 1000 --patience 20 --seed 42 \
        --id d4148af2649958c554afac0d997666c083cd806ff6e119eedd44a6ea585ecb27 \
        --no_use_gpu --num_workers 0 --checkpoints "00_final_model/ours"

环境变量（仅在模型构造时读取，控制超参）:
    HP__DROPOUT       dropout
    HP__HIDDEN_DIM    attention pool hidden dim
    HP__SEQ_LEN       positional embedding max_len
    HP__D_MODEL       d_model inside model
"""
import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M2SFormerATPSelfRegulationSCP2 minimal runner')

    # basic
    parser.add_argument('--task_name', type=str, required=True, default='classification',
                        help='only classification is supported in this minimal package')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='1=train+test, 0=test only')
    parser.add_argument('--model_id', type=str, required=True, default='test',
                        help='dataset name used to locate .ts files under root_path')
    parser.add_argument('--model', type=str, required=True, default='M2SFormerATPSelfRegulationSCP2',
                        help='model name; must match a file under models/')

    # data
    parser.add_argument('--data', type=str, required=True, default='UEA', help='only UEA is supported')
    parser.add_argument('--root_path', type=str, default='./dataset/',
                        help='dir containing <model_id>_TRAIN.ts / _TEST.ts')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='root dir for model checkpoints')

    # forecasting-shape params (kept for setting-string compatibility with the parent repo)
    parser.add_argument('--seq_len', type=int, default=96, help='overridden at runtime by dataset max_seq_len')
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=False)

    # model
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--use_norm', type=int, default=1)

    # Mamba-style params (kept for setting-string compatibility)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--d_conv', type=int, default=4)

    # optimization
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_type', type=str, default='cuda')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    # augmentation (default 0 -> not used; only kept for API compatibility)
    parser.add_argument('--augmentation_ratio', type=int, default=0)

    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1)
    parser.add_argument(
        '--id',
        type=str,
        default=datetime.now().strftime("%Y%m%d%H%M%S"),
        help='identifier appended to the setting string; must match checkpoint dir suffix for test mode'
    )

    args = parser.parse_args()

    # seed
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # device
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    elif torch.backends.mps.is_available() and args.use_gpu:
        args.device = torch.device("mps")
        print('Using MPS')
    else:
        args.device = torch.device("cpu")
        print('Using CPU')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.task_name != 'classification':
        raise ValueError(f'this minimal runner only supports classification, got task_name={args.task_name}')
    if args.data != 'UEA':
        raise ValueError(f'this minimal runner only supports UEA dataset, got data={args.data}')

    from exp.exp_classification import Exp_Classification
    Exp = Exp_Classification

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
            setting = setting + str(args.id)
            os.environ['TRAIN_START'] = str(time.time_ns())
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            os.environ['TRAIN_END'] = str(time.time_ns())
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
    else:
        exp = Exp(args)
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)
        setting = setting + str(args.id)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.use_gpu:
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
