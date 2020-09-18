# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/10/28 下午6:09

from model_define.model_resnet import fresnet50_v3, fresnet100_v3, fresnet34_v3
from collections import OrderedDict
from model_define.MobileFaceNet import MobileFaceNet
from model_define.model import MobileFaceNet_sor, ResNet34, MobileFaceNet_y2
from model_define.MobileNetV3 import MobileNetV3_Large
from model_define.resnet50_imagenet import resnet_50
from model_define.mobilefacenet_y2_ljt.mobilefacenet_big import MobileFaceNet_y2_ljt
from model_define.shufflefacenet_v2_ljt.ShuffleFaceNetV2 import ShuffleFaceNetV2
import torch


def load_state_dict(args):

    if args.model == 'mobilefacenet':
        model = MobileFaceNet_sor(args.embedding_size)

    elif args.model == 'resnet34':
        model = fresnet34_v3((112,112))
        args.lr = 0.001

    elif args.model == 'mobilefacenet_y2':
        model = MobileFaceNet_y2(args.embedding_size)

    elif args.model == 'resnet50':
        model = fresnet50_v3()

    elif args.model == 'resnet50_imagenet':
        model = resnet_50()

    elif args.model == 'resnet100':
        model = fresnet100_v3()

    elif args.model == 'mobilefacenet_lzc':
        model = MobileFaceNet(128, (5, 5))

    elif args.model == 'mobilenetv3':
        model = MobileNetV3_Large(4)

    elif args.model == 'resnet34_lzc':
        model = fresnet34_v3((80, 80))

    elif args.model == 'mobilefacenet_y2_ljt':
        model = MobileFaceNet_y2_ljt()

    elif args.model == 'shufflefacenet_v2_ljt':
        model = ShuffleFaceNetV2(512, 2.0, (144, 122))

    else:
        print('不支持此模型剪枝！')

    print('load {}\'s checkpoint'.format(args.model))

    state_dict = torch.load(args.best_model_path, map_location=args.device)
    if args.from_data_parallel:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)

    return model
