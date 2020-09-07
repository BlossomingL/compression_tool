from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch
from collections import OrderedDict
import math
import torch.nn.functional as F


class L2Norm(Module):
    def forward(self, input):
        return F.normalize(input)


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Conv_block_no_bn(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block_no_bn, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel,
                           groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MobileFaceNet(Module):
    def __init__(self, embedding_size, conv6_kernel=(7, 7), p=1, drop_p=0.75, num_classes=4, input_channel=3, fl=False):
        super(MobileFaceNet, self).__init__()
        # Focal Loss
        self.fl = fl
        self.embedding_size = embedding_size
        self.conv1 = Conv_block(input_channel, int(64*p), kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(int(64*p), int(64*p), kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=int(64*p))
        self.conv_23 = Depth_Wise(int(64*p), int(64*p), kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=int(128*p))
        self.conv_3 = Residual(int(64*p), num_block=4, groups=int(128*p), kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(int(64*p), int(128*p), kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=int(256*p))
        self.conv_4 = Residual(int(128*p), num_block=6, groups=int(256*p), kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(int(128*p), int(128*p), kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=int(512*p))
        self.conv_5 = Residual(int(128*p), num_block=2, groups=int(256*p), kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(int(128*p), int(512*p), kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(int(512*p), int(512*p), groups=int(512*p), kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(int(512*p), embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.drop = torch.nn.Dropout(p=drop_p)
        self.prob = Linear(embedding_size, num_classes, bias=False)

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

        if self.fl:
            out = out.view(out.size(0), -1)
            return out

        else:

            out = self.conv_6_flatten(out)
            # out = self.linear(out)
            if self.embedding_size != 512:
                out = self.linear(out)
            out = self.bn(out)
            out = self.drop(out)
            out = self.prob(out)

            return out


if __name__ == '__main__':
    model = MobileFaceNet(128, (5, 5))
    pretrained = torch.load('/home/user1/linx/program/LightFaceNet/work_space/models/model_train_best/2019-09-25-12'
                            '-21_LiveBody_le_0.2_80x80_fake-20190924-train-data_live-0923_MobileFaceNet-d128-k-5-5'
                            '-c4_pytorch_iter_13000.pth', map_location='cpu')

    # model = torch.nn.DataParallel(model)
    new_state_dict = OrderedDict()
    for k, v in pretrained.items():
        print(k, v.shape)
        new_state_dict[k[7:]] = v
    model.load_state_dict(new_state_dict)
    x = torch.rand([2, 3, 80, 80])
    model.cuda()
    x = x.to('cuda')
    out = model(x)
    print(out)