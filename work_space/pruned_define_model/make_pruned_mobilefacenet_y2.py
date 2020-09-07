# [64, 20, 20, 64, 26, 26, 64, 90, 90, 64, 64, 64, 64, 64, 64, 64, 52, 52, 64, 64, 64, 64, 52, 52, 64, 64, 64, 64, 77, 77, 77, 64, 231, 231, 128, 128, 128, 128, 154, 154, 128, 77, 77, 128, 77, 77, 128, 52, 52, 128, 52, 52, 128, 52, 52, 128, 77, 77, 128, 52, 52, 128, 103, 103, 128, 26, 26, 128, 77, 77, 128, 52, 52, 128, 77, 77, 128, 77, 77, 128, 103, 103, 128, 410, 410, 128, 26, 26, 128, 26, 26, 128, 26, 26, 128, 26, 26, 128, 512, 512]
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, ReLU6, Dropout2d, Dropout, \
    AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn
from collections import namedtuple
import math
import pdb
from torchsummary import summary
from functools import reduce
from torchvision.models import resnet34, resnet50


# #################################  Original Arcface Model #########################################################
class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SEModule(Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# #################################  MobileFaceNet #############################################################
class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        # print(in_c, out_c, groups)
        self.conv = Conv2d(in_c, out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
        # self.prelu = ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
     def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, se=False, use_cbam=False, use_sge=False):
        super(Depth_Wise, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        # 普通1x1卷积
        self.conv = Conv_block(c1_in, c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        # 深度卷积
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
        # self.se = SEModule(out_c) if se else None
        # self.cbam = CBAM( out_c, 16 ) if use_cbam else None
        # self.sge = SpatialGroupEnhance(64) if use_sge else None

     def forward(self, x):
        short_cut = x

        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        # if self.se is not None:
        #     x = self.se(x)
        # if self.cbam is not None:
        #     x = self.cbam(x)
        # if self.sge is not None:
        #     x = self.sge(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self, c1, c2, c3, num_block, groups, residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se=False, use_cbam=False, use_sge=False):
        super(Residual, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            modules.append(Depth_Wise(c1_tuple, c2_tuple, c3_tuple, residual=residual, kernel=kernel, padding=padding, stride=stride, groups=groups, se=se, use_cbam=use_cbam, use_sge=use_sge))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


# ################################# mobilefacenet_y2 #############################################################
class MobileFaceNet_y2(Module):
    def __init__(self, keep, embedding_size):
        super(MobileFaceNet_y2, self).__init__()
        self.block_info = {'conv2_dw': 2, 'conv_23': 1, 'conv_3': 8, 'conv_34': 1, 'conv_4': 16, 'conv_45': 1,
                           'conv_5': 4}

        self.conv1 = Conv_block(3, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        i = 0
        c1, c2, c3 = [], [], []
        for _ in range(2):
            c1.append((keep[i], keep[i + 1]))
            c2.append((keep[i + 1], keep[i + 2]))
            c3.append((keep[i + 2], keep[i + 3]))
            i += 3
        self.conv2_dw = Residual(c1, c2, c3, num_block=2, groups=keep[i - 2 * 3], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        # c1, c2, c3 = [], [], []
        # for _ in range(8):
        #     c1.append((keep[i], keep[i + 1]))
        #     c2.append((keep[i + 1], keep[i + 2]))
        #     c3.append((keep[i + 2], keep[i + 3]))
        #     i += 3

        c1, c2, c3 = [], [], []
        for _ in range(1):
            c1.append((keep[i], keep[i + 1]))
            c2.append((keep[i + 1], keep[i + 2]))
            c3.append((keep[i + 2], keep[i + 3]))
            i += 3
        self.conv_23 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[i - 3])

        c1, c2, c3 = [], [], []
        for _ in range(8):
            c1.append((keep[i], keep[i + 1]))
            c2.append((keep[i + 1], keep[i + 2]))
            c3.append((keep[i + 2], keep[i + 3]))
            i += 3
        self.conv_3 = Residual(c1, c2, c3, num_block=8, groups=keep[i - 8 * 3], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1, c2, c3 = [], [], []
        for _ in range(1):
            c1.append((keep[i], keep[i + 1]))
            c2.append((keep[i + 1], keep[i + 2]))
            c3.append((keep[i + 2], keep[i + 3]))
            i += 3
        self.conv_34 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[i - 3])

        c1, c2, c3 = [], [], []
        for _ in range(16):
            c1.append((keep[i], keep[i + 1]))
            c2.append((keep[i + 1], keep[i + 2]))
            c3.append((keep[i + 2], keep[i + 3]))
            i += 3
        self.conv_4 = Residual(c1, c2, c3, num_block=16, groups=keep[i - 16 * 3], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1, c2, c3 = [], [], []
        for _ in range(1):
            c1.append((keep[i], keep[i + 1]))
            c2.append((keep[i + 1], keep[i + 2]))
            c3.append((keep[i + 2], keep[i + 3]))
            i += 3
        self.conv_45 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[i - 3])

        c1, c2, c3 = [], [], []
        for _ in range(4):
            c1.append((keep[i], keep[i + 1]))
            c2.append((keep[i + 1], keep[i + 2]))
            c3.append((keep[i + 2], keep[i + 3]))
            i += 3
        self.conv_5 = Residual(c1, c2, c3, num_block=4, groups=keep[i - 4 * 3], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(keep[i], keep[i + 1], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(keep[i + 1], keep[i + 2], groups=keep[i + 1], kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)


def Pruned_MobileFaceNet_y2(embeddings):
    keep = [64, 20, 20, 64, 26, 26, 64, 90, 90, 64, 64, 64, 64, 64, 64, 64, 52, 52, 64, 64, 64, 64, 52, 52, 64, 64, 64, 64, 77, 77, 64, 77, 77, 64, 231, 231, 128, 128, 128, 128, 154, 154, 128, 77, 77, 128, 77, 77, 128, 52, 52, 128, 52, 52, 128, 52, 52, 128, 77, 77, 128, 52, 52, 128, 103, 103, 128, 26, 26, 128, 77, 77, 128, 52, 52, 128, 77, 77, 128, 77, 77, 128, 103, 103, 128, 410, 410, 128, 26, 26, 128, 26, 26, 128, 26, 26, 128, 26, 26, 128, 512, 512]
    return MobileFaceNet_y2(keep, embeddings)


# ##============================================================================================================
if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = ResNet34().to(device)
    # summary(model, (3, 112, 112))
    keep = [64, 20, 20, 64, 26, 26, 64, 90, 90, 64, 64, 64, 64, 64, 64, 64, 52, 52, 64, 64, 64, 64, 52, 52, 64, 64, 64, 64, 77, 77, 64, 77, 77, 64, 231, 231, 128, 128, 128, 128, 154, 154, 128, 77, 77, 128, 77, 77, 128, 52, 52, 128, 52, 52, 128, 52, 52, 128, 77, 77, 128, 52, 52, 128, 103, 103, 128, 26, 26, 128, 77, 77, 128, 52, 52, 128, 77, 77, 128, 77, 77, 128, 103, 103, 128, 410, 410, 128, 26, 26, 128, 26, 26, 128, 26, 26, 128, 26, 26, 128, 512, 512]
    # print(len(keep))
    model = Pruned_MobileFaceNet_y2(keep, 512)
    state_dict = torch.load('/home/yeluyue/lz/program/compression_tool/work_space/pruned_model/model_mobilefacenet_y2.pt')
    # for k, v in state_dict.items():
    #     print(k, v.shape)
