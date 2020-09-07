# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/10/30 下午7:37
import torch.nn as nn
import math
from torchvision.models.resnet import ResNet, Bottleneck
from torch.nn import Module
import torch.nn.functional as F
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.ReLU()
        # self.relu = nn.PReLU(planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
        self.conv2 = conv3x3(out_c1, out_c2, stride)
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


# suitable for input size 112x112
class ResNet(nn.Module):

    # def __init__(self, block, layers, num_classes=1000):
    def __init__(self, block, layers, keep, embedding_size=512):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(3, eps=2e-5, momentum=0.9)
        self.conv1 = nn.Conv2d(3, keep[0], kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(keep[0], eps=2e-5, momentum=0.9)
        self.relu = nn.PReLU(keep[0])
        # self.relu = nn.ReLU()
        # self.relu = nn.PReLU(64)

        #In caffe ceil_mode is always True
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
        self.bn3 = nn.BatchNorm2d(keep[-1], eps=2e-5, momentum=0.9)
        self.dp = nn.Dropout2d(p=0.4)
        # ks = ((input_size[0]+15)//16, (input_size[1]+15)//16)
        # print("kernel size{}".format(ks))
        ks = 7
        self.fc1 = nn.Conv2d(keep[-1], embedding_size, ks, bias=False)
        self.ft1 = Flatten()
        self.bn4 = nn.BatchNorm1d(embedding_size, eps=2e-5, momentum=0.9)
        # self.fc = nn.
        self.l2 = L2Norm()   #add l2 norm layer

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

        layers = []
        if level == 1:
            stride_c1 = [(keep[0], keep[1])]
            stride_c2 = [(keep[1], keep[2])]
            c1 = [(keep[2], keep[3]), (keep[4], keep[5])]
            c2 = [(keep[3], keep[4]), (keep[5], keep[6])]
            downsample = nn.Sequential(
                nn.Conv2d(keep[0], keep[2],
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(keep[2], eps=2e-5, momentum=0.9),
            )
        elif level == 2:
            stride_c1 = [(keep[6], keep[7])]
            stride_c2 = [(keep[7], keep[8])]
            c1 = [(keep[8], keep[9]), (keep[10], keep[11]), (keep[12], keep[13]), (keep[14], keep[15]),
                  (keep[16], keep[17]), (keep[18], keep[19]), (keep[20], keep[21]), (keep[22], keep[23]),
                  (keep[24], keep[25]), (keep[26], keep[27]), (keep[28], keep[29]), (keep[30], keep[31])]
            c2 = [(keep[9], keep[10]), (keep[11], keep[12]), (keep[13], keep[14]), (keep[15], keep[16]),
                  (keep[17], keep[18]), (keep[19], keep[20]), (keep[21], keep[22]), (keep[23], keep[24]),
                  (keep[25], keep[26]), (keep[27], keep[28]), (keep[29], keep[30]), (keep[31], keep[32])]
            downsample = nn.Sequential(
                nn.Conv2d(keep[6], keep[8],
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(keep[8], eps=2e-5, momentum=0.9),
            )
        elif level == 3:
            stride_c1 = [(keep[32], keep[33])]
            stride_c2 = [(keep[33], keep[34])]
            c1 = [(keep[34], keep[35]), (keep[36], keep[37]), (keep[38], keep[39]), (keep[40], keep[41]), (keep[42], keep[43]), (keep[44], keep[45]),
                  (keep[46], keep[47]), (keep[48], keep[49]), (keep[50], keep[51]), (keep[52], keep[53]), (keep[54], keep[55]), (keep[56], keep[57]),
                  (keep[58], keep[59]), (keep[60], keep[61]), (keep[62], keep[63]), (keep[64], keep[65]), (keep[66], keep[67]), (keep[68], keep[69]),
                  (keep[70], keep[71]), (keep[72], keep[73]), (keep[74], keep[75]), (keep[76], keep[77]), (keep[78], keep[79]), (keep[80], keep[81]),
                  (keep[82], keep[83]), (keep[84], keep[85]), (keep[86], keep[87]), (keep[88], keep[89]), (keep[90], keep[91])]

            c2 = [(keep[35], keep[36]), (keep[37], keep[38]), (keep[39], keep[40]), (keep[41], keep[42]), (keep[43], keep[44]), (keep[45], keep[46]),
                  (keep[47], keep[48]), (keep[49], keep[50]), (keep[51], keep[52]), (keep[53], keep[54]), (keep[55], keep[56]), (keep[57], keep[58]),
                  (keep[59], keep[60]), (keep[61], keep[62]), (keep[63], keep[64]), (keep[65], keep[66]), (keep[67], keep[68]), (keep[69], keep[70]),
                  (keep[71], keep[72]), (keep[73], keep[74]), (keep[75], keep[76]), (keep[77], keep[78]), (keep[79], keep[80]), (keep[81], keep[82]),
                  (keep[83], keep[84]), (keep[85], keep[86]), (keep[87], keep[88]), (keep[89], keep[90]), (keep[91], keep[92])]
            downsample = nn.Sequential(
                nn.Conv2d(keep[32], keep[34],
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(keep[34], eps=2e-5, momentum=0.9),
            )
        else:
            stride_c1 = [(keep[92], keep[93])]
            stride_c2 = [(keep[93], keep[94])]
            c1 = [(keep[94], keep[95]), (keep[96], keep[97])]
            c2 = [(keep[95], keep[96]), (keep[97], keep[98])]
            downsample = nn.Sequential(
                nn.Conv2d(keep[92], keep[94],
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(keep[94], eps=2e-5, momentum=0.9),
            )

        layers.append(block(stride_c1[0], stride_c2[0], stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            c_in = c1[i-1]
            c_out = c2[i-1]
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
        x = self.dp(x)
        x = self.fc1(x)
        x = self.ft1(x)
        x = self.bn4(x)
        x = self.l2(x)
        # x = self.avgpool(x)
        # x = self.flattern(x)
        # # x = self.fc(x)
        # x = self.l2(x)  #add l2 norm layer

        return x


def fresnet100_v3(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    keep_dict = {'74.2M': [32, 32, 26, 20, 26, 32, 26, 77, 64, 52, 64, 39, 64, 39, 64, 64, 64, 52, 64, 52, 64, 52, 64, 64, 64, 52, 64, 52, 64, 52, 64, 39, 64, 205, 128, 103, 128, 103, 128, 154, 128, 77, 128, 128, 128, 103, 128, 103, 128, 103, 128, 103, 128, 103, 128, 103, 128, 128, 128, 77, 128, 103, 128, 77, 128, 77, 128, 103, 128, 103, 128, 103, 128, 103, 128, 77, 128, 77, 128, 77, 128, 103, 128, 103, 128, 103, 128, 103, 128, 103, 128, 103, 128, 410, 256, 205, 256, 154, 256],
                 '101M': [64, 7, 64, 7, 64, 7, 64, 13, 128, 13, 128, 13, 128, 13,
                          128, 13, 128, 13, 128, 13, 128, 13, 128, 13, 128, 13, 128,
                          13, 128, 13, 128, 13, 128, 52, 256, 26, 256, 26, 256, 26,
                          256, 26, 256, 26, 256, 26, 256, 26, 256, 26, 256, 26, 256,
                          26, 256, 26, 256, 26, 256, 26, 256, 26, 256, 26, 256, 26,
                          256, 26, 256, 26, 256, 26, 256, 26, 256, 26, 256, 26, 256,
                          26, 256, 26, 256, 26, 256, 26, 256, 26, 256, 26, 256, 26,
                          256, 52, 512, 52, 512, 52, 512],
                 '131.6M': [64, 26, 64, 20, 64, 26, 64, 77, 128, 39, 128, 39, 128,
                            39, 128, 52, 128, 64, 128, 52, 128, 52, 128, 52, 128,
                            52, 128, 39, 128, 39, 128, 39, 128, 205, 256, 103, 256,
                            103, 256, 103, 256, 77, 256, 103, 256, 77, 256, 77, 256,
                            77, 256, 103, 256, 103, 256, 103, 256, 103, 256, 77, 256,
                            77, 256, 77, 256, 77, 256, 77, 256, 77, 256, 103, 256, 77,
                            256, 77, 256, 77, 256, 77, 256, 103, 256, 103, 256, 77, 256,
                            77, 256, 77, 256, 103, 256, 359, 512, 205, 512, 154, 512],
                 '202.7M': [64, 58, 64, 45, 64, 45, 64, 103, 128, 90, 128, 77, 128, 77,
                         128, 103, 128, 77, 128, 103, 128, 116, 128, 103, 128, 116,
                         128, 90, 128, 116, 128, 77, 128, 231, 256, 154, 256, 180, 256,
                         231, 256, 103, 256, 205, 256, 231, 256, 154, 256, 103, 256, 205,
                         256, 180, 256, 205, 256, 231, 256, 231, 256, 154, 256, 231, 256,
                         103, 256, 205, 256, 231, 256, 231, 256, 180, 256, 154, 256, 103,
                         256, 103, 256, 205, 256, 231, 256, 231, 256, 231, 256, 180, 256,
                         180, 256, 410, 512, 461, 512, 205, 512]}

    model = ResNet(BasicBlock_v3, [3, 13, 30, 3], keep_dict['74.2M'], **kwargs)
    model.load_state_dict(torch.load('/home/user1/linx/program/pruning_tool/work_space/pruned_model/model_resnet100.pt'))

    return model


if __name__ == '__main__':
    fresnet100_v3()
