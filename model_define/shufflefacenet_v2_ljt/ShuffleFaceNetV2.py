# -*- coding: utf-8 -*-
# @Time : 20-5-28 下午1:59
# @Author : ljt
# @Company : Minivision
# @File : ShuffleFaceNetV2.py
# @Software : PyCharm
# ShuffleFaceNet Version convert model with convert3 method

import torch.nn as nn
# from .blocks import ShuffleV2Block
import math
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module
import torch.nn.functional as F
import torch
# from ...common_utility import L2Norm, Flatten, GDC, get_shuffle_ave_pooling_size


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride, use_se=False):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(mid_channels),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.PReLU(outputs),
        ]
        # if use_se:
        #     branch_main.append(SEModule(outputs, 4))
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.PReLU(inp),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x = self.channel_shuffle(old_x)
            x_projs = torch.split(x, x.shape[1] // 2, dim=1)
            # x_projs = torch.split(old_x, old_x.shape[1] // 2, dim=1)
            return torch.cat((x_projs[0], self.branch_main(x_projs[1])), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    # def channel_shuffle(self, x):
    #     batchsize, num_channels, height, width = x.data.size()
    #
    #     assert (num_channels % 4 == 0)
    #     x = x.reshape(batchsize * num_channels // 2, 2, height,width)
    #     x = x.permute(1, 0, 2, 3)
    #     x = x.reshape(2, -1, num_channels // 2, height, width)
    #
    #     return x[0], x[1]

    def channel_shuffle(self, x):
            n, c, h, w = x.data.shape[0], x.data.shape[1],\
                         x.data.shape[2], x.data.shape[3]
            groups = 2
            x = x.view(n, groups, c // groups, h, w)
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(n, c, h, w)
            return x


class L2Norm(Module):
    def forward(self, input):
        return F.normalize(input)


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        # for onnx model convert
        # batch_size = np.array(input.size(0))
        # batch_size.astype(dtype=np.int32)
        # return input.view(batch_size, 512)


class GDC(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(GDC, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel,
                              groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


def get_shuffle_ave_pooling_size(height, width, using_pool=False):
    first_batch_num = 2
    if using_pool:
        first_batch_num = 3

    size1 = Get_Conv_kernel(height, width, (3, 3), (2, 2), (0, 0), first_batch_num)
    # print(size1)
    size2 = Get_Conv_kernel(size1[0], size1[1], (2, 2), (2, 2), (0, 0), 2)
    return size2


def Get_Conv_kernel(height, width, kernel, stride, padding, rpt_num):
    conv_h = height
    conv_w = width
    for _ in range(rpt_num):
        conv_h = math.ceil((conv_h - kernel[0] + 2 * padding[0]) / stride[0] + 1)
        conv_w = math.ceil((conv_w - kernel[1] + 2 * padding[1]) / stride[1] + 1)
        print(conv_h, conv_w)
    return (int(conv_h), int(conv_w))


class ShuffleFaceNetV2(nn.Module):
    def __init__(self, num_classes, width_multiplier, input_size, use_se=False):
        super(ShuffleFaceNetV2, self).__init__()

        gdc_size = get_shuffle_ave_pooling_size(input_size[0], input_size[1])

        print('gdc kernel size is ', gdc_size)

        self.use_se = use_se
        # self.use_3d = use_3d

        self.stage_repeats = [4, 8, 4]
        self.model_size = str(width_multiplier) + 'x'
        if self.model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif self.model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif self.model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif self.model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.PReLU(input_channel),
        )

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(ShuffleV2Block(input_channel, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=2,
                                                        use_se=self.use_se))
                else:
                    self.features.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                        mid_channels=output_channel // 2, ksize=3, stride=1,
                                                        use_se=self.use_se))

                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            nn.PReLU(self.stage_out_channels[-1])
        )

        self.gdc = GDC(self.stage_out_channels[-1], self.stage_out_channels[-1], kernel=gdc_size,
                       groups=self.stage_out_channels[-1])

        if self.model_size == '2.0x':
            self.dropout = nn.Dropout(0.2)
        # if self.use_se:
        #     self.se_last = SEModule(self.stage_out_channels[-1], 4)

        self.classifier = nn.Conv2d(self.stage_out_channels[-1], num_classes, 1)

        # if self.use_3d:
        #     self.classifier_3d = nn.Conv2d(self.stage_out_channels[-1], 59, 1)

        self.flatten = Flatten()
        self.bn = nn.BatchNorm1d(num_classes)
        self.l2 = L2Norm()

    def forward(self, x, use_pyamp=False):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.gdc(x)

        if self.use_se:
            x = self.se_last(x)

        x = self.classifier(x)
        # if self.model_size == '2.0x':
        #     x = self.dropout(x)
        x = self.flatten(x)
        x = self.bn(x)
        x = self.l2(x)
        return x


if __name__ == '__main__':
    model = ShuffleFaceNetV2(512, 2.0, (144, 122))
    state_dict = torch.load('/home/linx/model/ljt/2020-09-15-10-53_CombineMargin-ljt914-m0.9m0.4m0.15s64_le_re_0.4_144x122_2020-07-30-Full-CLEAN-0803-2-MIDDLE-30_ShuffleFaceNetA-2.0-d512_model_iter-76608_TYLG-0.7319_XCHoldClean-0.8198_BusIDPhoto-0.7310-noamp.pth')
    model.load_state_dict(state_dict)


