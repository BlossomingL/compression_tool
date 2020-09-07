# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/10/9 下午2:17

import torch
import os
from collections import OrderedDict
from copy import deepcopy
import distiller
from distiller.scheduler import CompressionScheduler
import numpy as np
from model_define.load_state_dict import load_state_dict
import time
from test_module.test_on_diverse_dataset import test
from datetime import datetime


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def perform_sensitivity_analysis(model, net_params, sparsities, args):

    sensitivities = OrderedDict()
    print('测试原模型精度')
    accuracy = test(args, model)

    print('原模型精度为:{}'.format(accuracy))

    if args.fpgm:
        print('即将采用几何中位数剪枝产生折线图')
    conv_dict = {}
    if args.hrank:
        print('即将采用HRank剪枝')
        cnt = 1
        layer_name = 'conv1.conv.weight'
        conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
        cnt += 1
        for key, value in model.block_info.items():
            if value == 1:
                layer_name = '{}.conv.conv.weight'.format(key)
                conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
                cnt += 1
                layer_name = '{}.conv_dw.conv.weight'.format(key)
                conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
                cnt += 1
                layer_name = '{}.project.conv.weight'.format(key)
                conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
                cnt += 1
            else:
                for j in range(value):
                    layer_name = '{}.model.{}.conv.conv.weight'.format(key, j)
                    conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
                    cnt += 1
                    layer_name = '{}.model.{}.conv_dw.conv.weight'.format(key, j)
                    conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
                    cnt += 1
                    layer_name = '{}.model.{}.project.conv.weight'.format(key, j)
                    conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
                    cnt += 1
        layer_name = 'conv_6_sep.conv.weight'
        conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
        cnt += 1
        layer_name = 'conv_6_dw.conv.weight'
        conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
        cnt += 1
        print(len(conv_dict))

    for param_name in net_params:
        if model.state_dict()[param_name].dim() not in [4]:
            continue

        model_cpy = deepcopy(model)

        sensitivity = OrderedDict()

        # 对每一层循环剪枝并测试精度(从0.05->0.95)
        for sparsity_level in sparsities:

            sparsity_level = float(sparsity_level)

            print(param_name, sparsity_level)

            pruner = distiller.pruning.L1RankedStructureParameterPruner("sensitivity",
                                                                        group_type="Filters",
                                                                        desired_sparsity=sparsity_level,
                                                                        weights=param_name)

            policy = distiller.PruningPolicy(pruner, pruner_args=None)
            scheduler = CompressionScheduler(model_cpy)
            scheduler.add_policy(policy, epochs=[0])

            scheduler.on_epoch_begin(0, fpgm=args.fpgm, HRank=args.hrank, conv_index=conv_dict)

            scheduler.mask_all_weights()

            accuracy = test(args, model_cpy)

            print('剪枝{}后的精度为：{}'.format(sparsity_level, accuracy))

            sensitivity[sparsity_level] = (accuracy, 0, 0)
            sensitivities[param_name] = sensitivity

    return sensitivities


def sensitivity_analysis(args):

    model = load_state_dict(args)
    model.eval()
    model.cuda()

    sensitivities = np.arange(0.0, 0.95, 0.1)
    which_params = [param_name for param_name, _ in model.named_parameters()]

    start_time = time.time()

    sensitivity = perform_sensitivity_analysis(model, which_params, sensitivities, args)

    end_time = time.time()
    print('剪枝敏感度分析总共耗时{}h'.format((end_time - start_time) / 3600))
    # distiller.sensitivities_to_png(sensitivity, 'work_space/sensitivity_data/sensitivity_{}.png'.format(args.model))
    distiller.sensitivities_to_csv(sensitivity, os.path.join(args.sensitivity_csv_path, 'sensitivity_{}_{}.csv'.format(args.model, get_time())))






