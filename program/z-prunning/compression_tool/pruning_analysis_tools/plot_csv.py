# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/10/9 上午10:03

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot(sensitivity, save_root):
    """
    画每个layer的折线图
    :param layer_name: layer名
    :param layer: 每个layer中的各个weight的敏感值
    :param save_root: 折线图存放路径
    :return:
    """
    i = 0
    for k, v in sensitivity.items():

        if i % 7 == 6:
            plt.ylabel('top1')
            plt.xlabel('sparsity')
            plt.title(str(i-6) + '-' + str(i+1) + ' Pruning Sensitivity')
            plt.legend(loc='lower center',
                       ncol=2, mode="expand", borderaxespad=0.)
            plt.savefig(save_root + '/' + str(i-6) + '-' + str(i+1) + '.png', format='png')
            plt.close()

        sense = v
        name = k
        sparsities = np.arange(0, 0.95, 0.1)
        plt.plot(sparsities, sense, label=name)
        i += 1


def plot_csv(csv_name, save_root):
    """
    plot sensitivity
    :param csv_name: csv文件名
    :param save_root: 图片保存路径
    :return:
    """
    data = pd.read_csv(csv_name)
    sensitivity = {}

    for x in data.values:
        sensitivity[x[0]] = []

    for x in data.values:
        sensitivity[x[0]].append(x[2])

    plot(sensitivity, save_root)


if __name__ == '__main__':
    plot_csv('/home/linx/program/z-prunning/compression_tool/work_space/sensitivity_data/sensitivity_resnet50_imagenet_2020-09-02-12-37.csv', '/home/linx/program/z-prunning/compression_tool/work_space/sensitivity_data')
