import torch
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module
from ..attention_module.se_uint import SEModule, SEModuleChannel


class ChannelShuffle(Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        x = x.reshape(x.shape[0], self.groups, x.shape[1] // self.groups, x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        return x


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


###############################################Depth_Wise################################################################################

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


# correct shuffle version
class Depth_Wise_Shufflev2(Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1,
                 shuffle_group=3):
        super(Depth_Wise_Shufflev2, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1), groups=shuffle_group)
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1), groups=shuffle_group)
        self.residual = residual
        self.shuffle = ChannelShuffle(shuffle_group)

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Depth_Wise_SE(Depth_Wise):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1,
                 se_reduct=10):
        super(Depth_Wise_SE, self).__init__(in_c, out_c, residual, kernel, stride, padding, groups)
        self.se_module = SEModule(out_c, se_reduct)

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            x = self.se_module(x)
            output = short_cut + x
        else:
            output = x
        return output


class Depth_Wise_SEChannel(Depth_Wise_SE):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1,
                 se_reduct=10):
        super(Depth_Wise_SEChannel, self).__init__(in_c, out_c, residual, kernel, stride, padding, groups)
        self.se_module = SEModuleChannel(out_c, se_reduct)


###############################################Residual################################################################################
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


class ResidualSE(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se_reduct=10):
        super(ResidualSE, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise_SE(c, c, residual=True, kernel=kernel,
                                         padding=padding, stride=stride, groups=groups, se_reduct=se_reduct))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class ResidualSEChannel(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), se_reduct=10):
        super(ResidualSEChannel, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise_SEChannel(c, c, residual=True, kernel=kernel,
                                                padding=padding, stride=stride, groups=groups, se_reduct=se_reduct))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


# correct shuffle version
class ResidualShufflev2(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), shuffle_group=3):
        super(ResidualShufflev2, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise_Shufflev2(c, c, residual=True, kernel=kernel, padding=padding,
                                                stride=stride, groups=groups, shuffle_group=shuffle_group))
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


################################################MixNet#########################################################
def _split_channels(num_channels, num_groups):
    split = [num_channels // num_groups for _ in range(num_groups)]
    split[0] += num_channels - sum(split)
    return split


class MixConv2d(Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1):
        super(MixConv2d, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_c, num_groups)
        out_splits = _split_channels(out_c, num_groups)
        self.splits = in_splits
        for i in range(num_groups):
            self.add_module(
                str(i), Conv2d(in_splits[i], out_splits[i], kernel_size[i], stride=stride,
                               padding=kernel_size[i] // 2, groups=out_splits[i], bias=False))

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x) for x, c in zip(x_split, self._modules.values())]
        x = torch.cat(x_out, 1)
        return x


class MixConv_block(Module):
    def __init__(self, in_c, out_c, kernel=3, stride=(1, 1)):
        super(MixConv_block, self).__init__()
        self.conv = MixConv2d(in_c, out_c, kernel, stride)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class MixNetBlock(Module):
    def __init__(self, in_c, out_c, groups=1, residual=False, kernel=(3, 3), stride=1, expand=1, padding=(1, 1),
                 se_reduct=1):
        super(MixNetBlock, self).__init__()
        self.expand = (expand != 1)
        self.se = (se_reduct != 1)
        self.residual = residual
        self.expand_channels = in_c * expand
        if self.expand:
            self.conv_pw = Conv_block(in_c, out_c=self.expand_channels, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = MixConv_block(self.expand_channels, self.expand_channels, kernel, stride)
        if self.se:
            self.squeeze_excite = Sequential(SEModule(groups, se_reduct))
        self.project = Linear_block(self.expand_channels, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))

    def forward(self, x):
        if self.residual:
            short_cut = x
        if self.expand:
            x = self.conv_pw(x)
        x = self.conv_dw(x)
        if self.se:
            x = self.squeeze_excite(x)
        x = self.project(x)
        if self.residual:
            out = short_cut + x
        else:
            out = x
        return out
