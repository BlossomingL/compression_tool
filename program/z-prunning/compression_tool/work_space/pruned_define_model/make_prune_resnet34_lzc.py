# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/11/6 上午11:00
import torch.nn as nn
import math
from torchvision.models.resnet import ResNet, Bottleneck
from torch.nn import Module
import torch.nn.functional as F
from collections import OrderedDict
import math
import torch
'''
Net work's common utility
'''


class L2Norm(Module):
    def forward(self, input):
        return F.normalize(input)


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


__all__ = ['ResNet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class L2Norm(Module):
    def forward(self, input):
        return F.normalize(input)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        in_c1, out_c1 = c_in
        in_c2, out_c2 = c_out
        self.conv1 = conv3x3(in_c1, out_c1, stride)
        self.bn1 = nn.BatchNorm2d(out_c1)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.ReLU()
        # self.relu = nn.PReLU(planes)

        self.conv2 = conv3x3(in_c2, out_c2)
        self.bn2 = nn.BatchNorm2d(out_c2)
        self.downsample = downsample
        self.stride = stride
        self.relu_res = nn.ReLU(inplace=True)
        # self.relu_res = nn.ReLU()
        # self.relu_res = nn.PReLU(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu_res(out)

        return out


class BasicBlock_v3(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1, downsample=None):
        super(BasicBlock_v3, self).__init__()
        in_c1, out_c1 = c_in
        in_c2, out_c2 = c_out
        self.bn1 = nn.BatchNorm2d(in_c1, eps=2e-5, momentum=0.9)
        self.conv1 = conv3x3(in_c1, out_c1, 1)
        # self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(out_c1, eps=2e-5, momentum=0.9)
        self.relu = nn.PReLU(out_c1)
        self.conv2 = conv3x3(in_c2, out_c2, stride)
        self.bn3 = nn.BatchNorm2d(out_c2, eps=2e-5, momentum=0.9)
        self.downsample = downsample
        self.stride = stride
        # self.relu_res = nn.ReLU(inplace=True)
        # self.relu_res = nn.ReLU()
        # self.relu_res = nn.PReLU(planes)

    def forward(self, x):
        residual = x
        out = self.bn1(x)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class ResNet_34(nn.Module):

    # def __init__(self, block, layers, num_classes=1000):
    def __init__(self, keep, block, layers, input_size=(112, 112), embedding_size=128, input_channel=3, num_classes=4):
        self.inplanes = 64
        super(ResNet_34, self).__init__()
        self.bn1 = nn.BatchNorm2d(input_channel, eps=2e-5, momentum=0.9)
        self.conv1 = nn.Conv2d(input_channel, keep[0], kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(keep[0], eps=2e-5, momentum=0.9)
        self.relu = nn.PReLU(keep[0])
        # self.relu = nn.ReLU()
        # self.relu = nn.PReLU(64)

        # In caffe ceil_mode is always True
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(keep, 1, block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(keep, 2, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(keep, 3, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(keep, 4, block, 512, layers[3], stride=2)
        # fea_size = Get_Conv_Size(input_size/4, input_size/4, (3,3),(2,2),(1,1),4)
        # print("feasize{}".format(fea_size))
        # self.avgpool = nn.AvgPool2d(fea_size, stride=1)
        # self.flattern = Flatten()
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bn3 = nn.BatchNorm2d(512*block.expansion, eps=2e-5, momentum=0.9)
        self.dp = nn.Dropout2d(p=0.4)
        ks = ((input_size[0]+15)//16, (input_size[1]+15)//16)
        print("kernel size{}".format(ks))
        self.fc1 = nn.Conv2d(512*block.expansion, embedding_size, ks, bias=False)
        self.ft1 = Flatten()
        self.bn4 = nn.BatchNorm1d(embedding_size, eps=2e-5, momentum=0.9)
        # self.fc = nn.
        # self.l2 = L2Norm()   # add l2 norm layer

        self.drop = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(embedding_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, keep, level, block, planes, blocks, stride=1):
        downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion: let first resnet-block has branch
        if level == 1:
            stride_c1 = [(keep[0], keep[1])]
            stride_c2 = [(keep[1], keep[2])]
            c1 = [(keep[2], keep[3]), (keep[4], keep[5])]
            c2 = [(keep[3], keep[4]), (keep[5], keep[6])]
            downsample = None
        elif level == 2:
            stride_c1 = [(keep[6], keep[7])]
            stride_c2 = [(keep[7], keep[8])]
            c1 = [(keep[8], keep[9]), (keep[10], keep[11]), (keep[12], keep[13])]
            c2 = [(keep[9], keep[10]), (keep[11], keep[12]), (keep[13], keep[14])]
            downsample = nn.Sequential(
                nn.Conv2d(keep[6], keep[8], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(keep[8], eps=2e-5, momentum=0.9)
            )
        elif level == 3:
            stride_c1 = [(keep[14], keep[15])]
            stride_c2 = [(keep[15], keep[16])]
            c1 = [(keep[16], keep[17]), (keep[18], keep[19]), (keep[20], keep[21]), (keep[22], keep[23]),
                  (keep[24], keep[25])]
            c2 = [(keep[17], keep[18]), (keep[19], keep[20]), (keep[21], keep[22]), (keep[23], keep[24]),
                  (keep[25], keep[26])]
            downsample = nn.Sequential(
                nn.Conv2d(keep[14], keep[16], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(keep[16], eps=2e-5, momentum=0.9)
            )
        else:
            stride_c1 = [(keep[26], keep[27])]
            stride_c2 = [(keep[27], keep[28])]
            c1 = [(keep[28], keep[29]), (keep[30], keep[31])]
            c2 = [(keep[29], keep[30]), (keep[31], keep[32])]
            downsample = nn.Sequential(
                nn.Conv2d(keep[26], keep[28], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(keep[28], eps=2e-5, momentum=0.9)
            )

        layers = []
        layers.append(block(stride_c1[0], stride_c2[0], stride, downsample))
        for i in range(1, blocks):
            c_in = c1[i - 1]
            c_out = c2[i - 1]
            layers.append(block(c_in, c_out, 1, None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn3(x)
        # x = self.dp(x)
        x = self.fc1(x)
        x = self.ft1(x)

        x = self.bn4(x)
        # x = self.l2(x)
        # x = self.avgpool(x)
        # x = self.flattern(x)
        # # x = self.fc(x)
        # x = self.l2(x)  #add l2 norm layer
        x = self.drop(x)
        x = self.fc2(x)

        return x


def fresnet34_v3(keep, input_size, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_34(keep, BasicBlock_v3, [3, 4, 6, 3], input_size, **kwargs, embedding_size=128)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


if __name__ == '__main__':
    keep_dict = {'28.8M': [64, 26, 64, 20, 64, 26, 64, 64, 128,
                            64, 128, 26, 128, 26, 128, 103, 256,
                            77, 256, 77, 256, 52, 256, 52, 256,
                            52, 256, 103, 512, 154, 512, 103, 512],
                 '42.4M': [64, 32, 64, 32, 64, 39, 64, 90, 128,
                           77, 128, 64, 128, 52, 128, 128, 256, 103,
                           256, 103, 256, 103, 256, 103, 256, 103,
                           256, 205, 512, 205, 512, 205, 512]}
    model = fresnet34_v3(keep_dict['42.4M'], (80, 80))
    model.load_state_dict(torch.load('/home/user1/桌面/resnet34_fpgm/model_resnet34_0.2.pt'))
