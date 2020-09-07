# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/10/23 上午10:07
import torch.nn as nn
import torch


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        in_c1, out_c1 = c_in
        in_c2, out_c2 = c_out
        self.conv1 = conv3x3(in_c1, out_c1, stride)
        self.bn1 = norm_layer(out_c1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_c1, out_c2)
        self.bn2 = norm_layer(out_c2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, keep, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, keep[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(keep[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(keep, 1, block, 64, layers[0])
        self.layer2 = self._make_layer(keep, 2, block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(keep, 3, block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(keep, 4, block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(512, 512)

        self.backbone = nn.Sequential(self.conv1,
                                      self.bn1,
                                      self.relu,
                                      self.maxpool,
                                      self.layer1,
                                      self.layer2,
                                      self.layer3,
                                      self.layer4,
                                      self.avgpool
                                      )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, keep, level, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
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
                conv1x1(keep[6], keep[8], stride),
                norm_layer(keep[8]),
            )
        elif level == 3:
            stride_c1 = [(keep[14], keep[15])]
            stride_c2 = [(keep[15], keep[16])]
            c1 = [(keep[16], keep[17]), (keep[18], keep[19]), (keep[20], keep[21]), (keep[22], keep[23]),
                  (keep[24], keep[25])]
            c2 = [(keep[17], keep[18]), (keep[19], keep[20]), (keep[21], keep[22]), (keep[23], keep[24]),
                  (keep[25], keep[26])]
            downsample = nn.Sequential(
                conv1x1(keep[14], keep[16], stride),
                norm_layer(keep[16]),
            )
        else:
            stride_c1 = [(keep[26], keep[27])]
            stride_c2 = [(keep[27], keep[28])]
            c1 = [(keep[28], keep[29]), (keep[30], keep[31])]
            c2 = [(keep[29], keep[30]), (keep[31], keep[32])]
            downsample = nn.Sequential(
                conv1x1(keep[26], keep[28], stride),
                norm_layer(keep[28]),
            )

        layers = []
        layers.append(block(stride_c1[0], stride_c2[0], stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        for i in range(1, blocks):
            c_in = c1[i - 1]
            c_out = c2[i - 1]
            layers.append(block(c_in, c_out, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        #
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        #
        # x = self.avgpool(x)
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)

        return l2_norm(x)


class ResNet34(nn.Module):
    def __init__(self, keep):
        super(ResNet34, self).__init__()
        resnet = ResNet(keep, BasicBlock, [3, 4, 6, 3])
        self.backbone = nn.Sequential(resnet.conv1,
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


if __name__ == '__main__':
    keep_dict = {'10M': [64, 26, 64, 26, 64, 7, 64, 26,
                         128, 13, 128, 13, 128, 13, 128,
                         52, 256, 26, 256, 26, 256, 26, 256,
                         26, 256, 26, 256, 52, 512, 52, 512, 52, 512],
                 '15M': [64, 32, 64, 39, 64, 7, 64, 39,
                         128, 13, 128, 13, 128, 13, 128, 103,
                         256, 26, 256, 26, 256, 26, 256, 26, 256,
                         26, 256, 154, 512, 52, 512, 52, 512],
                 '20M': [64, 32, 64, 45, 64, 13, 64, 52,
                         128, 26, 128, 13, 128, 13, 128, 128,
                         256, 26, 256, 26, 256, 26, 256, 26,
                         256, 26, 256, 154, 512, 154, 512, 52, 512],
                 '25M': [64, 39, 64, 45, 64, 13, 64, 64, 128, 26,
                         128, 13, 128, 26, 128, 154, 256, 26, 256,
                         26, 256, 26, 256, 26, 256, 26, 256, 205,
                         512, 256, 512, 52, 512]}
    model = ResNet34(keep_dict['25M'])
    model.load_state_dict(torch.load('/home/user1/桌面/resnet34/24.9M/model_resnet34_0.5.pt'))

