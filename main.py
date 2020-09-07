# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/11/12 下午1:38
import argparse
from sensitivity_analysis import sensitivity_analysis
from pruning import prune
from quantization import quantization
from datetime import datetime
import os
from train_module.train_with_insight_face import face_learner


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def main():
    parser = argparse.ArgumentParser(description='prune face recognition model')

    # 剪枝
    parser.add_argument('--lr', default=0.01, type=float, help='retrain学习率, 一般为训练时的1/10')
    parser.add_argument('--weight_decay', default=4e-5, type=float, help='学习率衰减')
    parser.add_argument('--save_pruned_model_root', default='work_space/models/pruned_model/', help='剪枝模型定义和文件保存文件夹')
    parser.add_argument('--momentum', default=0.9, type=float)

    parser.add_argument('--epoch', default=30, type=int, help='剪枝后重训练多少个epoch')
    parser.add_argument('--head_path', default=None, help='训练头')
    parser.add_argument('--device', default='cuda:0')

    parser.add_argument('--print_freq', type=int, default=1, help='每隔多少次打印准确度信息')
    parser.add_argument('--save_model_pt', default=False, action='store_true', help='是否保存pt文件')

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--pruned_save_model_path', default='work_space/pruned_model', help='剪枝后模型保存路径')
    parser.add_argument('--sensitivity_csv_path', default='work_space/sensitivity_data', help='剪枝敏感度分析后的csv文件保存路径')

    # 每次运行需要确定以下参数
    parser.add_argument('--mode', default=None, choices=['prune', 'quantization', 'test', 'sa', 'finetune'], help='prune表示仅仅剪枝,quantization表示量化'
                                                                                               'sa表示sensitivity analysis,'
                                                                                                  'finetune表示剪枝并finetune')

    parser.add_argument('--best_model_path', default=None, help='已经训练好的最好的模型文件路径，准备用来剪枝')
    parser.add_argument('--test_root_path', default=None, help='测试集root路径')
    parser.add_argument('--img_list_label_path', default=None, help='测试集pair list路径')
    parser.add_argument('--model', default=None,
                        choices=['mobilefacenet', 'resnet34', 'mobilefacenet_y2', 'resnet50', 'resnet100',
                                 'mobilefacenet_lzc', 'mobilenetv3', 'resnet34_lzc', 'resnet50_imagenet'], help='对哪个模型剪枝')

    parser.add_argument('--is_save', default=False, action='store_true', help='是否保存模型文件')

    parser.add_argument('--from_data_parallel', action='store_true', default=False, help='模型是否来自多卡训练')

    parser.add_argument('--data_source', choices=['lfw', 'company', 'company_zkx'], default='None',
                        help='测试时使用哪个测试集, company->zy的resnet50和resnet100, company_zkx->zkx的mobilefacenet_y2')

    parser.add_argument('--fpgm', action='store_true', default=False, help='是否使用几何中位数剪枝')
    parser.add_argument('--hrank', action='store_true', default=False, help='是否使用HRank剪枝')
    parser.add_argument('--rank_path', default='./work_space/rank_conv/', help='HRank配置文件')

    parser.add_argument('--yaml_path', default='yaml_file/auto_yaml.yaml', help='剪枝配置文件')

    parser.add_argument('--cal_flops_and_forward', default=False, action='store_true', help='是否测试flops和前向时间')

    parser.add_argument('--test_batch_size', type=int, default=256)

    # 下面是量化时需要确定的参数
    parser.add_argument('--quantize-mode', type=str, choices=['symmetric', 'asymmetric-signed', 'unsigned'], default='symmetric',
                        help='量化模式，将权重值映射到对称，有符号非对称和无符号非对称区间')

    parser.add_argument('--fp16', action='store_true', default=False, help='采用半精度量化，设置了此模式，上面的模式都会失效')

    parser.add_argument('--input_size', type=int, default=112, help='输出图片大小')

    parser.add_argument('--quantized_save_model_path', default='work_space/quantized_model', help='量化后模型保存路径')

    # finetune时所需参数
    parser.add_argument('--pruned_checkpoint', type=str, default=None, help='剪枝后的模型文件路径')
    parser.add_argument('--train_data_path', type=str, default=None, help='finetune所需训练集的路径')
    parser.add_argument('--milestones', type=str, default='12,15,18', help='规定在第几个epoch学习率下降')
    parser.add_argument('--train_batch_size', type=int, default=128, help='训练batch size')
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--work_path', type=str, default='work_space/finetune', help='训练过程产生的文件存放目录')
    parser.add_argument('--finetune_pruned_model', action='store_true', default=False, help='finetune 剪枝后的模型')

    args = parser.parse_args()

    if args.mode == 'prune':
        prune(args)

    elif args.mode == 'sa':
        sensitivity_analysis(args)

    elif args.mode == 'quantization':
        quantization(args)

    elif args.mode == 'finetune':
        args.work_path = os.path.join(args.work_path, get_time())
        os.mkdir(args.work_path)

        args.log_path = os.path.join(args.work_path, 'log')
        args.save_path = os.path.join(args.work_path, 'save')
        args.model_path = os.path.join(args.work_path, 'model')

        os.mkdir(args.log_path)
        os.mkdir(args.save_path)
        os.mkdir(args.model_path)

        args.log_path = os.path.join(args.log_path, get_time())
        args.milestones = list(map(int, args.milestones.split(',')))

        learner = face_learner(args)
        learner.train(args)


if __name__ == '__main__':
    main()
