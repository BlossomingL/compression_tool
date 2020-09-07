# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/11/1 上午10:06
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, ReLU6, Dropout2d, Dropout, \
    AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import OrderedDict


# #################################  Original Arcface Model #########################################################
class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


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
     def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, se=False, use_cbam=False, use_sge=False):
        super(Depth_Wise, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        # 普通1x1卷积
        self.conv = Conv_block(c1_in, c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        # 深度卷积
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_out, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

     def forward(self, x):
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


class MobileFaceNet_y2(Module):
    def __init__(self, keep, embedding_size):
        super(MobileFaceNet_y2, self).__init__()
        self.conv1 = Conv_block(3, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        c1 = [(keep[0], keep[1]), (keep[3], keep[4])]
        c2 = [(keep[1], keep[2]), (keep[4], keep[5])]
        c3 = [(keep[2], keep[3]), (keep[5], keep[6])]
        self.conv2_dw = Residual(c1, c2, c3, num_block=2, groups=64, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[6], keep[7])]
        c2 = [(keep[7], keep[8])]
        c3 = [(keep[8], keep[9])]
        self.conv_23 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        c1 = [(keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16]), (keep[18], keep[19]), (keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), (keep[30], keep[31])]
        c2 = [(keep[10], keep[11]), (keep[13], keep[14]), (keep[16], keep[17]), (keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), (keep[28], keep[29]), (keep[31], keep[32])]
        c3 = [(keep[11], keep[12]), (keep[14], keep[15]), (keep[17], keep[18]), (keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), (keep[29], keep[30]), (keep[32], keep[33])]
        self.conv_3 = Residual(c1, c2, c3, num_block=8, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[33], keep[34])]
        c2 = [(keep[34], keep[35])]
        c3 = [(keep[35], keep[36])]
        self.conv_34 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        c1 = [(keep[36], keep[37]), (keep[39], keep[40]), (keep[42], keep[43]), (keep[45], keep[46]), (keep[48], keep[49]), (keep[51], keep[52]), (keep[54], keep[55]), (keep[57], keep[58]), (keep[60], keep[61]), (keep[63], keep[64]), (keep[66], keep[67]), (keep[69], keep[70]), (keep[72], keep[73]), (keep[75], keep[76]), (keep[78], keep[79]), (keep[81], keep[82])]
        c2 = [(keep[37], keep[38]), (keep[40], keep[41]), (keep[43], keep[44]), (keep[46], keep[47]), (keep[49], keep[50]), (keep[52], keep[53]), (keep[55], keep[56]), (keep[58], keep[59]), (keep[61], keep[62]), (keep[64], keep[65]), (keep[67], keep[68]), (keep[70], keep[71]), (keep[73], keep[74]), (keep[76], keep[77]), (keep[79], keep[80]), (keep[82], keep[83])]
        c3 = [(keep[38], keep[39]), (keep[41], keep[42]), (keep[44], keep[45]), (keep[47], keep[48]), (keep[50], keep[51]), (keep[53], keep[54]), (keep[56], keep[57]), (keep[59], keep[60]), (keep[62], keep[63]), (keep[65], keep[66]), (keep[68], keep[69]), (keep[71], keep[72]), (keep[74], keep[75]), (keep[77], keep[78]), (keep[80], keep[81]), (keep[83], keep[84])]
        self.conv_4 = Residual(c1, c2, c3, num_block=16, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        c1 = [(keep[84], keep[85])]
        c2 = [(keep[85], keep[86])]
        c3 = [(keep[86], keep[87])]
        self.conv_45 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        c1 = [(keep[87], keep[88]), (keep[90], keep[91]), (keep[93], keep[94]), (keep[96], keep[97])]
        c2 = [(keep[88], keep[89]), (keep[91], keep[92]), (keep[94], keep[95]), (keep[97], keep[98])]
        c3 = [(keep[89], keep[90]), (keep[92], keep[93]), (keep[95], keep[96]), (keep[98], keep[99])]
        self.conv_5 = Residual(c1, c2, c3, num_block=4, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_6_sep = Conv_block(keep[99], keep[100], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(keep[100], keep[101], groups=keep[101], kernel=(7, 7), stride=(1, 1), padding=(0, 0))
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


if __name__ == '__main__':
    keep_dict = {'4M_low_precision': [64, 45, 45, 64, 32, 32, 64, 77, 77, 64, 64, 64, 64, 26, 26, 64, 64, 64, 64, 52, 52, 64, 39, 39, 64, 26, 26, 64, 39, 39, 64, 52, 52, 64, 205, 205, 128, 26, 26, 128, 52, 52, 128, 154, 154, 128, 128, 128, 128, 180, 180, 128, 77, 77, 128, 26, 26, 128, 52, 52, 128, 26, 26, 128, 103, 103, 128, 26, 26, 128, 52, 52, 128, 103, 103, 128, 128, 128, 128, 103, 103, 128, 77, 77, 128, 359, 359, 128, 26, 26, 128, 26, 26, 128, 26, 26, 128, 26, 26, 128, 512, 512],
                 '4M_high_precision': [64, 45, 45, 64, 45, 45, 64, 77, 77, 64, 39, 39, 64, 26, 26, 64, 77, 77, 64, 39, 39, 64, 26, 26, 64, 39, 39, 64, 64, 64, 64, 26, 26, 64, 154, 154, 128, 26, 26, 128, 52, 52, 128, 180, 180, 128, 154, 154, 128, 180, 180, 128, 103, 103, 128, 26, 26, 128, 26, 26, 128, 26, 26, 128, 52, 52, 128, 77, 77, 128, 26, 26, 128, 77, 77, 128, 154, 154, 128, 154, 154, 128, 77, 77, 128, 410, 410, 128, 77, 77, 128, 52, 52, 128, 26, 26, 128, 26, 26, 128, 512, 512]}
    model = MobileFaceNet_y2(keep_dict['4M_high_precision'], embedding_size=512)
    new_state_dict = OrderedDict()
    state_dict = torch.load('/home/user1/桌面/mobilefacenet_y2_fpgm/high_precision_0.8118/model_mobilefacenet_y2_0.721.pt')
    model.load_state_dict(state_dict)
