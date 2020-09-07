# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/10/10 下午2:21

import matplotlib
import torch
import distiller
import os
from commom_utils.utils import cal_flops, test_speed, get_time
from model_define.load_state_dict import load_state_dict
from test_module.test_on_diverse_dataset import test
import numpy as np
matplotlib.use('agg')


def prune(args):
    model = load_state_dict(args)
    model = model.cuda()
    epoch = 0

    # acc = test(args, model)
    # print('剪枝前acc为：{}'.format(acc))

    if args.fpgm:
        print('使用fpgm算法剪枝')
    if args.cal_flops_and_forward:
        flops, params = cal_flops(model, [1, 3, 112, 112])
        forward_time = test_speed(model)
        print('剪枝前前向时间为{}ms, flops={}, params={}'.format(forward_time, flops, params))
    conv_dict = {}

    if args.hrank:

        cnt = 1
        layer_name = 'conv1.conv.weight'
        conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
        cnt += 1
        for key, value in model.block_info.items():
            if value == 1:
                layer_name = '{}.conv.conv.weight'.format(key)
                conv_dict[layer_name] = np.load(args.rank_path+'rank_conv'+str(cnt)+'.npy')
                cnt += 1
                layer_name = '{}.conv_dw.conv.weight'.format(key)
                conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
                cnt += 1
                layer_name = '{}.project.conv.weight'.format(key)
                conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
                cnt += 1
            else:
                for j in range(value):
                    layer_name = '{}.model.{}.conv.conv.weight'.format(key,j)
                    conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
                    cnt += 1
                    layer_name = '{}.model.{}.conv_dw.conv.weight'.format(key,j)
                    conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
                    cnt += 1
                    layer_name = '{}.model.{}.project.conv.weight'.format(key,j)
                    conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
                    cnt += 1
        layer_name = 'conv_6_sep.conv.weight'
        conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
        cnt += 1
        layer_name = 'conv_6_dw.conv.weight'
        conv_dict[layer_name] = np.load(args.rank_path + 'rank_conv' + str(cnt) + '.npy')
        cnt += 1
        print(len(conv_dict))

    model.train()
    compression_scheduler = distiller.config.file_config(model, None, args.yaml_path)
    compression_scheduler.on_epoch_begin(epoch, fpgm=args.fpgm, HRank=args.hrank, conv_index=conv_dict)
    compression_scheduler.on_minibatch_begin(epoch, minibatch_id=0, minibatches_per_epoch=0)

    if args.save_model_pt:
        torch.save(model.state_dict(), os.path.join(args.pruned_save_model_path, 'model_{}.pt'.format(args.model)))
        print('模型已保存！')
        config_model_arr = []
        for k, v in model.state_dict().items():
            if len(v.shape) == 4 and k.find('downsample') == -1 and k.find('fc') == -1:
                config_model_arr.append(v.shape[0])
        print('网络每层out参数为：{}, 总共{}个参数'.format(config_model_arr, len(config_model_arr)))
        # print(model)
        acc = test(args, model)
        print('剪枝后acc为：{}'.format(acc))

        flops, params = cal_flops(model, [1, 3, 112, 112])
        forward_time = 0
        print('剪枝后前向时间为{}ms, flops={}, params={}'.format(forward_time, flops, params))



