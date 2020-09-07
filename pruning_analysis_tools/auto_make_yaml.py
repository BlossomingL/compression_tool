# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/10/21 下午5:02
import pandas as pd
import numpy as np
import yaml


class WInfo:
    def __init__(self, name, sparsity):
        self.name = name
        self.sparsity = sparsity


def create_yaml_dict(sparsity_dict, img_size):
    """
    给定一个形如{稀疏度：[权值名称], 稀疏度：[权值名称], .....}的字典, 返回即将写入yaml文件的字典
    :param sparsity_dict: 权值稀疏度字典
    :param img_size: 输入图像大小
    :return: 即将被写入yaml文件的字典
    """
    yaml_dict = {}
    yaml_dict['version'] = 1
    yaml_dict['pruners'] = {}
    yaml_dict['extensions'] = {'net_thinner':
                                   {'class': 'FilterRemover',
                                    'thinning_func_str': 'remove_filters',
                                    'arch': 'None',
                                    'dataset': str(img_size[0]) + 'x' + str(img_size[1])}}

    yaml_dict['policies'] = []
    pruner_dict = yaml_dict['pruners']
    policies_dict = yaml_dict['policies']
    for key, value in sparsity_dict.items():
        if len(value) > 0:
            pruner_name = 'filter_pruner_' + str(int(key * 100))
            pruner_dict[pruner_name] = {'class': "L1RankedStructureParameterPruner",
                                        'group_type': 'Filters',
                                        'desired_sparsity': float(key),
                                        'weights': value}

            policies_dict.append({'pruner': {'instance_name': pruner_name}, 'epochs': [int(0)]})

    policies_dict.append({'extension': {'instance_name': 'net_thinner'}, 'epochs': [int(0)]})
    return yaml_dict


def config_yaml(csv_name, expect_acc, mode='mobilefacenet', img_size=(80, 80)):
    """
    设置一个期望精度并根据csv文件生成yaml文件
    :param csv_name: csv文件路径
    :param expect_acc: 期望剪枝后得到的精度
    :param mode: 根据不同的网络提供不同的策略
    :param img_size: 输入图像大小
    :return:
    """
    data = pd.read_csv(csv_name)

    sparsity = np.arange(0, 0.95, step=0.1)

    if mode == 'mobilefacenet' or mode == 'mobilefacenet_y2' or mode == 'mobilefacenet_lzc':
        res_spa = create_sparsity2weight_facenet(sparsity, data, expect_acc)

    elif mode == 'resnet50' or mode == 'resnet34' or mode == 'resnet100' or mode == 'resnet34_lzc':
        print(mode)
        res_spa = create_sparsity2weight_resnet(sparsity, data, expect_acc, mode)

    elif mode == 'conv2':
        res_spa = create_sparsity2weight_resnet_conv2(sparsity, data, expect_acc, mode)

    elif mode == 'mobilenetv3':
        res_spa = create_sparsity2weight_mobilenetv3(sparsity, data, expect_acc)

    yaml_dict = create_yaml_dict(res_spa, img_size)

    with open('../yaml_file/auto_yaml.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(yaml_dict, f)


def create_sparsity2weight_resnet_conv2(sparsity, data, expect_acc, mode):
    sensitivity = {}

    for x in data.values:
        if (x[0].find('conv1') != -1 or x[0].find('conv2') != -1) and x[0].find('downsample') == -1:
            sensitivity[x[0]] = []

    res_spa = {}
    for x in sparsity:
        x = round(x, 1)
        res_spa[x] = []

    for x in data.values:
        if (x[0].find('conv1') != -1 or x[0].find('conv2') != -1) and x[0].find('downsample') == -1:
            sensitivity[x[0]].append(x[2])

    res = []
    for key, value in sensitivity.items():
        proximal_arr = find_min_index(abs(np.array(value[1:]) - np.array([expect_acc] * len(value[1:]))))

        best_index = proximal_arr[0] + 1
        for i in range(best_index + 1, 10):
            if value[i] >= value[proximal_arr[0] + 1]:
                best_index = i
        res.append(WInfo(key, sparsity[best_index]))
        # for i, x in enumerate(value[1:]):
        #     if x < expect_acc:
        #         pos = i
        #         break
        #
        # if pos == -1:
        #     res.append(WInfo(key, sparsity[9]))
        # elif pos >= 1:
        #     res.append(WInfo(key, sparsity[pos]))

    min_arr = [1, 1, 1, 1]
    for x in res:
        if x.name.find('layer1'):
            if x.name.find('conv2') != -1 and x.name.find('downsample') == -1:
                if x.sparsity < min_arr[0]:
                    min_arr[0] = x.sparsity
        elif x.name.find('layer2'):
            if x.name.find('conv2') != -1 and x.name.find('downsample') == -1:
                if x.sparsity < min_arr[1]:
                    min_arr[1] = x.sparsity
        elif x.name.name.find('layer3'):
            if x.name.find('conv2') != -1 and x.name.find('downsample') == -1:
                if x.sparsity < min_arr[2]:
                    min_arr[2] = x.sparsity
        elif x.name.find('layer4'):
            if x.name.find('conv2') != -1 and x.name.find('downsample') == -1:
                if x.sparsity < min_arr[3]:
                    min_arr[3] = x.sparsity

    for x in res:
        if x.name.find('layer1'):
            if x.name.find('conv2') != -1 and x.name.find('downsample') == -1:
                x.sparsity = min_arr[0]
        elif x.name.find('layer2'):
            if x.name.find('conv2') != -1 and x.name.find('downsample') == -1:
                x.sparsity = min_arr[1]
        elif x.name.find('layer3'):
            if x.name.find('conv2') != -1 and x.name.find('downsample') == -1:
                x.sparsity = min_arr[2]
        elif x.name.find('layer4'):
            if x.name.find('conv2') != -1 and x.name.find('downsample') == -1:
                x.sparsity = min_arr[3]

    res[0].sparsity = min_arr[0]

    for x in res:
        x.sparsity = round(x.sparsity, 1)
        res_spa[x.sparsity].append(x.name)

    # conv_arr = []
    # for x in data.values:
    #     if x[0].find('conv') != -1 and x[0].find('downsample') == -1:
    #         conv_arr.append(x[0])

    return res_spa


def create_sparsity2weight_resnet(sparsity, data, expect_acc, mode):
    sensitivity = {}

    for x in data.values:
        if x[0].find('conv1') != -1:
            sensitivity[x[0]] = []

    res_spa = {}
    for x in sparsity:
        x = round(x, 1)
        res_spa[x] = []

    for x in data.values:
        if x[0].find('conv1') != -1:
            sensitivity[x[0]].append(x[2])

    res = []
    for key, value in sensitivity.items():
        proximal_arr = find_min_index(abs(np.array(value[1:]) - np.array([expect_acc] * len(value[1:]))))

        best_index = proximal_arr[0] + 1
        for i in range(best_index + 1, 10):
            if value[i] >= value[proximal_arr[0] + 1]:
                best_index = i
        res.append(WInfo(key, sparsity[best_index]))
        # for i, x in enumerate(value[1:]):
        #     if x < expect_acc:
        #         pos = i
        #         break
        #
        # if pos == -1:
        #     res.append(WInfo(key, sparsity[9]))
        # elif pos >= 1:
        #     res.append(WInfo(key, sparsity[pos]))
    if mode == 'resnet50' or mode == 'resnet100' or mode == 'resnet34_lzc':
        res = res[1:]

    for x in res:
        x.sparsity = round(x.sparsity, 1)
        res_spa[x.sparsity].append(x.name)

    # conv_arr = []
    # for x in data.values:
    #     if x[0].find('conv') != -1 and x[0].find('downsample') == -1:
    #         conv_arr.append(x[0])

    return res_spa


def create_sparsity2weight_facenet(sparsity, data, expect_acc):

    # 初始化一个敏感度字典，键为权值名称，值为敏感度值（0.1～0.9对应的精度值）
    sensitivity = {}
    for x in data.values:
        sensitivity[x[0]] = []

    # 初始化一个敏感度字典，键为稀疏度（从0.1～0.9），值为相稀疏度对应的权值名称
    res_spa = {}
    for x in sparsity:
        x = round(x, 1)
        res_spa[x] = []

    for x in data.values:
        sensitivity[x[0]].append(x[2])

    res = []

    # 找到大于期望精度的最小值
    for key, value in sensitivity.items():
        proximal_arr = find_min_index(abs(np.array(value[1:]) - np.array([expect_acc] * len(value[1:]))))

        best_index = proximal_arr[0] + 1
        for i in range(best_index + 1, 10):
            if value[i] >= value[proximal_arr[0] + 1]:
                best_index = i
        res.append(WInfo(key, sparsity[best_index]))

    # 稀疏度固定为一位小数
    for x in res:
        x.sparsity = round(x.sparsity, 1)

    res_temp = []
    flag = True
    project_seg = []
    # 最后两层不剪枝
    res.remove(res[-1])
    res.remove(res[-1])
    # 带deep-wise残差块的剪第一层需要先看后面的deep-wise层
    for i, winfo in enumerate(res):
        if winfo.name.find('conv_dw') != -1:
            replace_winfo_name = winfo.name.replace('conv_dw', 'conv')
            for winfo1 in res:
                if winfo1.name == replace_winfo_name:
                    winfo1.sparsity = min(winfo.sparsity, winfo1.sparsity)
                    res_temp.append(winfo1)
                    break

    #     elif winfo.name.find('project') != -1:
    #         # print(winfo.name)
    #         if flag:
    #             first_head = winfo.name.split('.')[0]
    #             min_temp = winfo.sparsity
    #             min_name = winfo.name
    #             project_seg.append(winfo)
    #             flag = False
    #             continue
    #         if (winfo.name.split('.')[0] == first_head or winfo.name.split('.')[0][-1] == first_head[-1]) and winfo.sparsity < min_temp:
    #             min_temp = winfo.sparsity
    #             min_name = winfo.name
    #             project_seg.append(winfo)
    #         elif winfo.name.split('.')[0] != first_head and winfo.name.split('.')[0][-1] != first_head[-1]:
    #             for x in project_seg:
    #                 x.sparsity = min_temp
    #                 res_temp.append(x)
    #             project_seg = []
    #             first_head = winfo.name.split('.')[0]
    #             min_temp = winfo.sparsity
    #             min_name = winfo.name
    #             project_seg.append(winfo)
    #         else:
    #             project_seg.append(winfo)
    #
    # res_temp.append(WInfo(min_name, min_temp))

    # 网络的第一个卷积层也剪枝
    # res_temp.append(res[0])

    for x in res_temp:
        res_spa[x.sparsity].append(x.name)

    return res_spa


def create_sparsity2weight_mobilenetv3(sparsity, data, expect_acc):

    sensitivity = {}
    for x in data.values:
        sensitivity[x[0]] = []
    res_spa = {}

    for x in sparsity:
        x = round(x, 1)
        res_spa[x] = []

    for x in data.values:
        sensitivity[x[0]].append(x[2])

    res = []

    # 找到大于期望精度的最小值
    for key, value in sensitivity.items():
        pos = -1

        for i, x in enumerate(value[1:]):
            if x < expect_acc:
                pos = i
                break

        if pos == -1:
            res.append(WInfo(key, sparsity[9]))
        elif pos >= 1:
            res.append(WInfo(key, sparsity[pos]))

    # 稀疏度固定为一位小数
    for x in res:
        x.sparsity = round(x.sparsity, 1)

    res_temp = []
    flag = True
    project_seg = []
    # 最后两层不剪枝
    res.remove(res[-1])
    res.remove(res[-1])
    for i, winfo in enumerate(res):
        if winfo.name.find('conv2') != -1:
            replace_winfo_name = winfo.name.replace('conv2', 'conv1')
            for winfo1 in res:
                if winfo1.name == replace_winfo_name:
                    winfo1.sparsity = min(winfo.sparsity, winfo1.sparsity)
                    res_temp.append(winfo1)
                    break
    # res_temp.append(res[0])

    for x in res_temp:
        res_spa[x.sparsity].append(x.name)
    return res_spa


def find_min_index(arr):
    min_index_arr = []
    min_value = 2
    for i, value in enumerate(arr):
        if value < min_value:
            min_value = value
            min_index = i

    for i, value in enumerate(arr):
        if value == arr[min_index]:
            min_index_arr.append(i)

    return min_index_arr


def main():
    # config_yaml('/home/user1/linx/program/LightFaceNet/work_space/sensitivity_data/mobilenetv3_0.6613/sensitivity_mobilenetv3.csv', 0.40, mode='mobilenetv3')
    config_yaml('/home/yeluyue/lz/program/compression_tool/work_space/sensitivity_data/sensitivity_mobilefacenet_y2_2020-09-04-13-08.csv', 0.985, mode='mobilefacenet', img_size=(112, 112))    # config_yaml('../work_space/sensitivity_data/mobilefacenet_y2_zkx_0.7889/fpgm/sensitivity_mobilefacenet_y2_2019-11-05-06-30.csv', 0.721, mode='mobilefacenet_y2', img_size=(112, 112))


if __name__ == '__main__':
    main()

