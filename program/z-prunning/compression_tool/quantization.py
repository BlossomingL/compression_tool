# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/11/12 下午1:38
import torch
from distiller.quantization.range_linear import PostTrainLinearQuantizer
from model_define.load_state_dict import load_state_dict
from test_module.test_on_diverse_dataset import test
from commom_utils.utils import get_time
import os


def quantization(args):
    model = load_state_dict(args)
    model = model.to(args.device)

    acc = test(args, model)
    print('量化前精度为:{}'.format(acc))

    quantizer = PostTrainLinearQuantizer(model, fp16=args.fp16)

    quantizer.prepare_model(torch.rand([1, 3, args.input_size, args.input_size]))

    torch.save(model.state_dict(), os.path.join(args.quantized_save_model_path, args.model + get_time() + '.pt'))
    print('模型已保存')

    acc = test(args, model)
    print('量化后精度为:{}'.format(acc))
