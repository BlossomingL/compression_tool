# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/10/29 下午8:19
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from MobileNetV3 import MobileNetV3_Large

writer = SummaryWriter('/home/user1/linx/program/LightFaceNet/work_space/log')

model = MobileNetV3_Large(4)

model = nn.DataParallel(model)

model.load_state_dict(torch.load('/home/user1/linx/program/LightFaceNet/work_space/models/model_train_best/2019-10-12'
                                 '-16-04_LiveBody_le_0.2_80x80_fake-20190924-train-data_live-0926_MobileNetv3Large'
                                 '-c4_pytorch_iter_14000.pth'))

for name, param in model.named_parameters():
    if len(param.shape) == 4:
        param = param.view(param.shape[0], -1)
        param = torch.norm(param, dim=1)
        print(param.shape)
        writer.add_histogram(name, param.clone().cpu().data.numpy())

# from scipy.spatial import distance
# import numpy as np
#
# a = np.array([[1, 2], [3, 4]])
# b = np.array([[2, 3], [4, 5]])
# print(distance.cdist(a, a, metric='euclidean'))
