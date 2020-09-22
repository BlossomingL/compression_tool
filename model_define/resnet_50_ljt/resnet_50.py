# -*- coding:utf-8 -*-
# author: linx
# datetime 2020/9/22 上午10:09
# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/10/21 下午7:31
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_v3, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes, eps=2e-5, momentum=0.9)
        self.conv1 = conv3x3(inplanes, planes, 1)
        # self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9)
        self.relu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9)
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


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BasicBlock_v3se(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_v3se, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes, eps=2e-5, momentum=0.9)
        self.conv1 = conv3x3(inplanes, planes, 1)
        # self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.ReLU()

        self.bn2 = nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9)
        self.relu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-5, momentum=0.9)
        self.downsample = downsample
        self.stride = stride
        self.semodule = SEModule(planes,16)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        se = self.semodule(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        se += residual

        return se


class GDC(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(GDC, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# suitable for input size 112x112
class ResNet(nn.Module):

    # def __init__(self, block, layers, num_classes=1000):
    def __init__(self, block, layers, embedding_size=512):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(3, eps=2e-5, momentum=0.9)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=2e-5, momentum=0.9)
        self.relu = nn.PReLU(64)
        # self.relu = nn.ReLU()
        # self.relu = nn.PReLU(64)

        # In caffe ceil_mode is always True
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # fea_size = Get_Conv_Size(input_size/4, input_size/4, (3,3),(2,2),(1,1),4)
        # print("feasize{}".format(fea_size))
        # self.avgpool = nn.AvgPool2d(fea_size, stride=1)
        # self.flattern = Flatten()
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bn3 = nn.BatchNorm2d(512*block.expansion, eps=2e-5, momentum=0.9)
        self.dp = nn.Dropout2d(p=0.4)
        # ks = ((input_size[0]+15)//16, (input_size[1]+15)//16)
        # print("kernel size{}".format(ks))
        # ks = 7
        self.fc1 = nn.Conv2d(512*block.expansion, embedding_size, kernel_size=(9, 8), bias=False)
        # self.fc1 = GDC(512 * block.expansion, embedding_size, kernel=(ks, ks), groups=512)
        self.ft1 = Flatten()
        self.bn4 = nn.BatchNorm1d(embedding_size, eps=2e-5, momentum=0.9)
        # self.fc = nn.
        self.l2 = L2Norm()   # add l2 norm layer

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion: let first resnet-block has branch

        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion, eps=2e-5, momentum=0.9),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None))

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


class ResNet_34(nn.Module):

    # def __init__(self, block, layers, num_classes=1000):
    def __init__(self, block, layers, input_size=(112, 112), embedding_size=128, input_channel=3, num_classes=4):
        self.inplanes = 64
        super(ResNet_34, self).__init__()
        self.bn1 = nn.BatchNorm2d(input_channel, eps=2e-5, momentum=0.9)
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=2e-5, momentum=0.9)
        self.relu = nn.PReLU(64)
        # self.relu = nn.ReLU()
        # self.relu = nn.PReLU(64)

        # In caffe ceil_mode is always True
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
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

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion: let first resnet-block has branch

        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion, eps=2e-5, momentum=0.9),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None))

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


class ResNet_224(nn.Module):

    # def __init__(self, block, layers, num_classes=1000):
    def __init__(self, block, layers, input_size=(112, 112), embedding_size=512):
        self.inplanes = 64
        super(ResNet_224, self).__init__()
        self.bn1 = nn.BatchNorm2d(3, eps=2e-5, momentum=0.9)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=2e-5, momentum=0.9)
        self.relu = nn.PReLU(64)
        # self.relu = nn.ReLU()
        # self.relu = nn.PReLU(64)

        # In caffe ceil_mode is always True
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # fea_size = Get_Conv_Size(input_size/4, input_size/4, (3,3),(2,2),(1,1),4)
        # print("feasize{}".format(fea_size))
        # self.avgpool = nn.AvgPool2d(fea_size, stride=1)
        # self.flattern = Flatten()
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bn3 = nn.BatchNorm2d(512*block.expansion, eps=2e-5, momentum=0.9)
        self.dp = nn.Dropout2d(p=0.4)
        ks = ((input_size[0]+31)//32, (input_size[1]+31)//32)
        # ks = 5
        print("kernel size{}".format(ks))
        self.fc1 = nn.Conv2d(512*block.expansion, embedding_size, ks, bias=False)
        self.ft1 = Flatten()
        self.bn4 = nn.BatchNorm1d(embedding_size, eps=2e-5, momentum=0.9)
        # self.fc = nn.
        self.l2 = L2Norm()   # add l2 norm layer

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # if stride != 1 or self.inplanes != planes * block.expansion: let first resnet-block has branch

        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion, eps=2e-5, momentum=0.9),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print("block shape",x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn3(x)
        x = self.dp(x)
        x = self.fc1(x)
        # print("fc", x.size())

        x = self.ft1(x)
        # print("ft", x.size())

        x = self.bn4(x)
        x = self.l2(x)
        # x = self.avgpool(x)
        # x = self.flattern(x)
        # # x = self.fc(x)
        # x = self.l2(x)  #add l2 norm layer

        return x


def fresnet50_v3_ljt(**kwargs):
    """
    Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock_v3, [3, 4, 14, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def fresnet34_v3(input_size, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if input_size[0] == 224:
        print("224 input: Initializing pooling input...")
        model = ResNet_224(BasicBlock_v3, [3, 4, 6, 3], input_size, **kwargs, embedding_size=128)
    else:
        model = ResNet(BasicBlock_v3, [3, 4, 6, 3], embedding_size=512)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def fresnet100_v3(**kwargs):
    """
    Constructs a ResNet-100 model.
    :param kwargs:
    :return:
    """
    model = ResNet(BasicBlock_v3, [3, 13, 30, 3], **kwargs)
    return model


if __name__ == '__main__':
    model = fresnet50_v3_ljt()
    state_dict = torch.load('/home/linx/model/ljt/2020-08-11-22-35_CombineMargin-ljt83-m0.9m0.4m0.15s64_le_re_0'
                            '.4_144x122_2020-07-30-Full-CLEAN-0803-2-MIDDLE-30_fResNet50v3cv-d512_model_iter'
                            '-76608_TYLG-0.8070_PadMaskYTBYGlassM280-0.9305_BusIDPhoto-0.6541.pth')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    model.load_state_dict(new_state_dict)
    # model_input = torch.rand([2, 3, 80, 80])
    # model.cuda()
    # model_input = model_input.to('cuda')
    # output = model(model_input)
    # print(output)
    # softmax_output = F.softmax(output)
    # print(softmax_output)
