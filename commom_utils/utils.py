# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/10/10 下午2:22

import time
from thop import profile
from tqdm import tqdm
from model_define.model import ResNet34
from model_define.resnet50_imagenet import resnet_50
from timeit import default_timer as timer
import torch
from datetime import datetime
from torchvision import transforms as trans
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import io


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def de_preprocess(tensor):
    return tensor*0.5 + 0.5


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def test_speed(model, device='gpu', test_time=10000):
    model.eval()
    inputs = torch.rand([1, 3, 112, 112])
    if device == 'gpu':
        model = model.to('cuda')
        inputs = inputs.to('cuda')
    else:
        model = model.to('cpu')
    print('Testing forward time,this may take a few minutes')
    start_time = timer()
    with torch.no_grad():
        for i in tqdm(range(test_time)):
            model(inputs)
    count = timer() - start_time
    forward_time = (count / test_time) * 1000
    print('平均forward时间为{}ms'.format(forward_time))
    return forward_time


def cal_flops(model, input_shape, device='gpu'):
    input_random = torch.rand(input_shape)
    if device == 'gpu':
        input_random = input_random.to('cuda')
        model = model.to('cuda')
    else:
        model = model.to('cpu')
    flops, params = profile(model, inputs=(input_random, ), verbose=False)
    return flops / (1024 * 1024 * 1024), params / (1024 * 1024)


hflip = trans.Compose([
            de_preprocess,
            trans.ToPILImage(),
            trans.functional.hflip,
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs


def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf


def main():
    # model = ResNet34()
    # state_dict = torch.load('/home/user1/linx/program/LightFaceNet/work_space/models/model_train_best'
    #                         '/resnet34_model_2019-10-12-19-20_accuracy:0.7216981_step:84816_lin.pth')
    # model.load_state_dict(state_dict)
    # test_speed(model)

    # model = torch.load('/home/user1/linx/program/LightFaceNet/work_space/models/pruned_model/model_resnet34.pkl')
    # state_dict = torch.load('/home/user1/linx/program/LightFaceNet/work_space/models/pruned_model'
    #                         '/resnet34_best_pruned_0.6556604.pt')
    # model.load_state_dict(state_dict)
    # test_speed(model)

    model = resnet_50()
    state_dict = torch.load('/home/linx/program/InsightFace_Pytorch/work_space/2020-08-20-10-32/models/model_accuracy'
                            ':0.9708333333333334_step:163760_best_acc_lfw.pth')
    model.load_state_dict(state_dict)
    print(cal_flops(model, (1, 3, 112, 112)))


if __name__ == '__main__':
    main()

