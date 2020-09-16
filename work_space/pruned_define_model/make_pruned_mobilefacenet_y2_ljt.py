'''
2019-09-29 mod by zkx add channel wise SE MobileFaceNety2 -> MobileFaceNetSEChannel_y2
'''
import math
from torch.nn import Linear, BatchNorm1d, Module, AdaptiveAvgPool2d
# from .network_elems import Linear_block, Conv_block, Depth_Wise, \
#     Residual, ResidualSE, ResidualShufflev2, ResidualSEChannel, BatchNorm2d
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module
import torch.nn.functional as F
import torch
# from .common_utility import L2Norm, Flatten, get_shuffle_ave_pooling_size
# import torch.cuda.amp as pyamp


def Get_Conv_kernel(height, width, kernel, stride, padding, rpt_num):
    conv_h = height
    conv_w = width
    for _ in range(rpt_num):
        conv_h = math.ceil((conv_h - kernel[0] + 2 * padding[0]) / stride[0] + 1)
        conv_w = math.ceil((conv_w - kernel[1] + 2 * padding[1]) / stride[1] + 1)
        print(conv_h, conv_w)
    return (int(conv_h), int(conv_w))


def get_shuffle_ave_pooling_size(height, width, using_pool=False):
    first_batch_num = 2
    if using_pool:
        first_batch_num = 3

    size1 = Get_Conv_kernel(height, width, (3, 3), (2, 2), (0, 0), first_batch_num)
    # print(size1)
    size2 = Get_Conv_kernel(size1[0], size1[1], (2, 2), (2, 2), (0, 0), 2)
    return size2


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


class Depth_Wise(Module):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
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


class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding,
                           bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
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


class Residual(Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []

        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            modules.append(
                Depth_Wise(c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MobileFaceNet_y2(Module):
    def __init__(self, keep, embedding_size, input_size=(112, 112)):
        super(MobileFaceNet_y2, self).__init__()
        ave_pool_size = get_shuffle_ave_pooling_size(input_size[0], input_size[1])
        self.conv1 = Conv_block(3, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))

        i = 0
        c1, c2, c3 = [], [], []
        for _ in range(2):
            c1.append((keep[i], keep[i + 1]))
            c2.append((keep[i + 1], keep[i + 2]))
            c3.append((keep[i + 2], keep[i + 3]))
            i += 3
        self.conv2_dw = Residual(c1, c2, c3, num_block=2, groups=keep[i - 2 * 3], kernel=(3, 3), stride=(1, 1), padding=(1, 1))

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
        self.conv_6_dw = Linear_block(keep[i + 1], keep[i + 2], groups=keep[i + 1], kernel=(int(ave_pool_size[0]), int(ave_pool_size[1])),
                                      stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.l2 = L2Norm()

    def forward(self, x, use_pyamp=False):
        # with pyamp.autocast(enabled=use_pyamp):
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
        out_6 = self.linear(out)
        out = self.bn(out_6)
        out = self.l2(out)
        return out


def Pruned_MobileFaceNet_y2_ljt(embeddings):
    keep =  {'BUSID': [64, 20, 20, 64, 26, 26, 64, 77, 77, 64, 13, 13, 64, 13, 13, 64, 77, 77, 64, 13, 13, 64, 39, 39, 64, 26, 26, 64, 13, 13, 64, 52, 52, 64, 231, 231, 128, 77, 77, 128, 103, 103, 128, 128, 128, 128, 154, 154, 128, 77, 77, 128, 103, 103, 128, 128, 128, 128, 52, 52, 128, 154, 154, 128, 103, 103, 128, 26, 26, 128, 26, 26, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 52, 52, 128, 461, 461, 128, 128, 128, 128, 26, 26, 128, 26, 26, 128, 26, 26, 128, 512, 512],
             'TYLG':[64, 7, 7, 64, 26, 26, 64, 52, 52, 64, 13, 13, 64, 13, 13, 64, 64, 64, 64, 13, 13, 64, 13, 13, 64, 26, 26, 64, 13, 13, 64, 26, 26, 64, 205, 205, 128, 77, 77, 128, 77, 77, 128, 154, 154, 128, 154, 154, 128, 77, 77, 128, 103, 103, 128, 205, 205, 128, 103, 103, 128, 26, 26, 128, 77, 77, 128, 103, 103, 128, 26, 26, 128, 103, 103, 128, 103, 103, 128, 103, 103, 128, 77, 77, 128, 359, 359, 128, 26, 26, 128, 26, 26, 128, 26, 26, 128, 52, 52, 128, 512, 512],
             'XCH': []}
    return MobileFaceNet_y2(keep['TYLG'], embeddings, (144, 122))
# def MobileFaceNet_y2_ljt():
#     model = MobileFaceNet_y2(512, (144, 122))
#     return model


if __name__ == '__main__':
    model = Pruned_MobileFaceNet_y2_ljt(512)
    state_dict = torch.load('/home/yeluyue/lz/program/compression_tool/work_space/pruned_model/model_mobilefacenet_y2_ljt_TYLG.pt')
    model.load_state_dict(state_dict)
