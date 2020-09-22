# -*- coding:utf-8 -*-
# author: linx
# datetime 2020/9/18 上午11:31
from model_define.shufflefacenet_v2_ljt.ShuffleFaceNetV2 import ShuffleFaceNetV2
import torch

state_dict = torch.load('/home/linx/model/ljt/2020-09-15-10-53_CombineMargin-ljt914-m0.9m0.4m0.15s64_le_re_0.4_144x122_2020-07-30-Full-CLEAN-0803-2-MIDDLE-30_ShuffleFaceNetA-2.0-d512_model_iter-76608_TYLG-0.7319_XCHoldClean-0.8198_BusIDPhoto-0.7310-noamp.pth')

for k, v in state_dict.items():
    print(k, v.shape)
    pass

net = ShuffleFaceNetV2(512, '2.0', (144, 122))
print(net)