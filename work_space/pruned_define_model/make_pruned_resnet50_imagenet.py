# -*- coding:utf-8 -*-
# author: linx
# datetime 2020/9/1 上午9:41
import torch.nn as nn
# from torch.cuda.amp import autocast, GradScaler
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
import math

# ================================= ResNet50 base on ImageNet ========================================================
norm_mean, norm_var = 1.0, 0.1


class L2Norm(Module):
    def forward(self, input):
        return F.normalize(input)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, c_in, c_out, stride=1, downsample=None, cp_rate=[0.], tmp_name=None):
        super(ResBottleneck, self).__init__()
        in_c1, in_c2, in_c3 = c_in
        out_c1, out_c2, out_c3 = c_out
        self.conv1 = nn.Conv2d(in_c1, out_c1, kernel_size=1, bias=False)
        self.conv1.cp_rate = cp_rate[0]
        self.conv1.tmp_name = tmp_name
        self.bn1 = nn.BatchNorm2d(out_c1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_c2, out_c2, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c2)
        self.conv2.cp_rate = cp_rate[1]
        self.conv2.tmp_name = tmp_name
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_c3, out_c3, kernel_size=1, bias=False)
        self.conv3.cp_rate = cp_rate[2]
        self.conv3.tmp_name = tmp_name

        self.bn3 = nn.BatchNorm2d(out_c3)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class Downsample(nn.Module):
    def __init__(self, downsample):
        super(Downsample, self).__init__()
        self.downsample = downsample

    def forward(self, x):
        out = self.downsample(x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, keep, covcfg=None, compress_rate=None):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.covcfg = covcfg
        self.compress_rate = compress_rate
        self.num_blocks = num_blocks

        self.conv1 = nn.Conv2d(3, keep[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1.cp_rate = compress_rate[0]
        self.conv1.tmp_name = 'conv1'
        self.bn1 = nn.BatchNorm2d(keep[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, keep, 0, num_blocks[0], stride=1,
                                       cp_rate=compress_rate[1:3*num_blocks[0]+2],
                                       tmp_name='layer1')
        self.layer2 = self._make_layer(block, keep, 9, num_blocks[1], stride=2,
                                       cp_rate=compress_rate[3*num_blocks[0]+2:3*num_blocks[0]+3*num_blocks[1]+3],
                                       tmp_name='layer2')
        self.layer3 = self._make_layer(block, keep, 21, num_blocks[2], stride=2,
                                       cp_rate=compress_rate[3*num_blocks[0]+3*num_blocks[1]+3:3*num_blocks[0]+3*num_blocks[1]+3*num_blocks[2]+4],
                                       tmp_name='layer3')
        self.layer4 = self._make_layer(block, keep, 39, num_blocks[3], stride=2,
                                       cp_rate=compress_rate[3*num_blocks[0]+3*num_blocks[1]+3*num_blocks[2]+4:],
                                       tmp_name='layer4')

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.bn2 = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(512 * block.expansion, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.l2_norm = L2Norm()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, keep, last_blocks, blocks, stride, cp_rate, tmp_name):
        layers = []

        i = last_blocks
        downsample = nn.Sequential(
            nn.Conv2d(keep[i], keep[i + 3],
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(keep[i + 3], eps=2e-5, momentum=0.9),
        )
        stride_c1 = [(keep[i], keep[i + 1], keep[i + 2])]
        stride_c2 = [(keep[i + 1], keep[i + 2], keep[i + 3])]
        i += 3
        c1 = []
        c2 = []
        for _ in range(1, blocks):
            c1.append((keep[i], keep[i + 1], keep[i + 2]))
            c2.append((keep[i + 1], keep[i + 2], keep[i + 3]))
            i += 3

        layers.append(block(stride_c1[0], stride_c2[0], stride, downsample, cp_rate, tmp_name))

        for i in range(1, blocks):
            c_in = c1[i - 1]
            c_out = c2[i - 1]
            layers.append(block(c_in, c_out, 1, None, cp_rate, tmp_name))

        return nn.Sequential(*layers)

    def forward(self, x, amp=False):
        # with autocast(enabled=amp):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # 256 x 56 x 56
        x = self.layer2(x)

        # 512 x 28 x 28
        x = self.layer3(x)

        # 1024 x 14 x 14
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.bn2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = self.bn3(x)
        x = self.l2_norm(x)

        return x


def resnet_50(compress_rate=[0] * 53):
    cov_cfg = [(3*i + 3) for i in range(3*3 + 1 + 4*3 + 1 + 6*3 + 1 + 3*3 + 1 + 1)]
    # fpgm
    # keep = [52, 20, 7, 256, 20, 7, 256, 20, 7, 256, 39,
    #         103, 512, 39, 103, 512, 39, 103, 512, 39,
    #         103, 512, 154, 205, 1024, 77, 205, 1024,
    #         128, 205, 1024, 180, 205, 1024, 103, 205,
    #         1024, 180, 205, 1024, 461, 410, 2048, 256,
    #         410, 2048, 154, 410, 2048]
    keep = [52, 20, 7, 256, 20, 7, 256, 20, 7, 256, 39,
            103, 512, 39, 103, 512, 52, 103, 512, 39,
            103, 512, 128, 205, 1024, 103, 205, 1024,
            103, 205, 1024, 180, 205, 1024, 103, 205,
            1024, 180, 205, 1024, 461, 410, 2048, 256,
            410, 2048, 205, 410, 2048]
    model = ResNet(ResBottleneck, [3, 4, 6, 3], keep, covcfg=cov_cfg, compress_rate=compress_rate)
    return model


# #################################  Arcface head #############################################################
def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Arcface(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def forward(self, embbedings, label, amp=False):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
        #         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask].float()

        if amp:
            output = (cos_theta * 1.0).float()  # a little bit hacky way to prevent in_place operation on cos_theta
        else:
            output = (cos_theta * 1.0)

        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output


if __name__ == '__main__':
    m = resnet_50()
    state_dict = torch.load('/home/linx/program/z-prunning/compression_tool/work_space/pruned_model'
                            '/model_resnet50_imagenet.pt')
    m.load_state_dict(state_dict)