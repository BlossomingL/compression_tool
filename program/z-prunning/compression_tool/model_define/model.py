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
     def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, se=False, use_cbam=False, use_sge=False):
        super(Depth_Wise, self).__init__()
        # 普通1x1卷积
        self.conv = Conv_block(in_c, groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        # 深度卷积
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
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
    def __init__(self, c_in, c_out, num_block, groups, residual=True, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se=False, use_cbam=False, use_sge=False):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c_in, c_out, residual=residual, kernel=kernel, padding=padding, stride=stride, groups=groups, se=se, use_cbam=use_cbam, use_sge=use_sge))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


# ################################# mobilefacenet_sor #############################################################
class MobileFaceNet_sor(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet_sor, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))

        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)

        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, 64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se=False)

        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, 128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se=False)

        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, 128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se=False)

        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        # GDConv 对大小为[512,7,7]的特征图进行全局深度卷积
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))

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
        # print(out.shape)
        out = self.conv_6_dw(out)
        # print(out.shape)
        out = self.conv_6_flatten(out)
        # print(out.shape)
        out = self.linear(out)

        out = self.bn(out)

        return l2_norm(out)


# ################################# mobilefacenet_21 #############################################################
class MobileFaceNet_21(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet_21, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(128, num_block=8, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(128, 256, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(256, num_block=5, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(256, 256, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(256, num_block=4, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(256, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
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


# ################################# mobilefacenet_23 #############################################################
class MobileFaceNet_23(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet_23, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Residual(64, num_block=4, groups=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_23 = Depth_Wise(64, 96, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=96)
        self.conv_3 = Residual(96, num_block=4, groups=96, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(96, 128, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=128)
        self.conv_4 = Residual(128, num_block=2, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 192, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=192)
        self.conv_5 = Residual(192, num_block=2, groups=192, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_56 = Depth_Wise(192, 256, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
        self.conv_6 = Residual(256, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_67 = Depth_Wise(256, 384, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=384)
        self.conv_7 = Residual(384, num_block=1, groups=384, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_8_sep = Conv_block(384, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_8_r = Residual(512, num_block=1, groups=512, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_8_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_8_flatten = Flatten()
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
        out = self.conv_56(out)
        out = self.conv_6(out)
        out = self.conv_67(out)
        out = self.conv_7(out)
        out = self.conv_8_sep(out)
        out = self.conv_8_r(out)
        out = self.conv_8_dw(out)
        out = self.conv_8_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return l2_norm(out)


# ################################# mobilefacenet_y2 #############################################################
class MobileFaceNet_y2(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet_y2, self).__init__()
        self.block_info = {'conv2_dw': 2, 'conv_23': 1, 'conv_3': 8, 'conv_34': 1, 'conv_4': 16, 'conv_45': 1,
                           'conv_5': 4}
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Residual(64, 64, num_block=2, groups=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, 64, num_block=8, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, 128, num_block=16, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, 128, num_block=4, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
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


# ################################# resnet34 #############################################################
class ResNet34(Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        resnet = resnet34(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.linear = nn.Linear(512, 512)

    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return l2_norm(x)


# ##============================================================================================================
if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = ResNet34().to(device)
    # summary(model, (3, 112, 112))
    model = resnet50()
    state_dict = torch.load('/home/user1/linx/program/LightFaceNet/work_space/models/model_train_best/2019-09-29-05'
                            '-31_SVGArcFace-O1-b0.4s40t1.1_fc_0.4_112x112_2019-09-27-Adult-padSY-Bus_fResNet50v3cv'
                            '-d512_pytorch_iter_360000.pth')
    for k, v in state_dict.items():
        print(k, v.shape)
