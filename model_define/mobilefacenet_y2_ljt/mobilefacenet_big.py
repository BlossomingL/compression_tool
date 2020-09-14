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
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(
                Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MobileFaceNet_y2(Module):
    def __init__(self, embedding_size, input_size=(112, 112)):
        super(MobileFaceNet_y2, self).__init__()
        ave_pool_size = get_shuffle_ave_pooling_size(input_size[0], input_size[1])
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Residual(64, num_block=2, groups=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=8, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=16, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=4, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(int(ave_pool_size[0]), int(ave_pool_size[1])),
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


def MobileFaceNet_y2_ljt():
    model = MobileFaceNet_y2(512, (144, 122))
    return model


if __name__ == '__main__':
    model = MobileFaceNet_y2(512, (144, 122))

    state_dict = torch.load('/home/linx/model/ljt/2020-08-23-08-09_CombineMargin-ljt83-m0.9m0.4m0.15s64_le_re_0'
                            '.4_144x122_2020-07-30-Full-CLEAN-0803-2-ID-INTRA-MIDDLE-30-INTER-90-HARD_MobileFaceNety2'
                            '-d512-k-9-8_model_iter-125993_TYLG-0.7520_PadMaskYTBYGlassM280-0.9104_BusIDPhoto-0.7489-noamp'
                            '.pth')

    model.load_state_dict(state_dict)
