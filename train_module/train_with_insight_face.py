# -*- coding:utf-8 -*-
# author: linx
# datetime 2020/9/1 上午9:19
import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from data.train_data.ms1m_10k_loader import get_train_loader, get_val_data
from work_space.pruned_define_model.make_pruned_resnet50_imagenet import resnet_50, Arcface, l2_norm
from work_space.pruned_define_model.make_pruned_mobilefacenet_y2 import Pruned_MobileFaceNet_y2
from work_space.pruned_define_model.make_pruned_resnet50 import pruned_fresnet50_v3
from data.train_data.verifacation import evaluate
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import torch.nn as nn

plt.switch_backend('agg')
from commom_utils.utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
from model_define.model import MobileFaceNet_y2
from model_define.model_resnet import fresnet50_v3


class face_learner(object):
    def __init__(self, args):
        if args.finetune_pruned_model:
            if args.model == 'resnet50_imagenet':
                self.model = resnet_50([0] * 53).to(args.device)
                print('pruned ResNet-50(ImageNet) model generated')

            elif args.model == 'mobilefacenet_y2':
                self.model = Pruned_MobileFaceNet_y2(args.embedding_size).to(args.device)
                print('pruned mobilefacenet_y2 mdoel generated')

            elif args.model == 'resnet50':
                self.model = pruned_fresnet50_v3().to(args.device)
                print('pruned ResNet-50(公司) model generated')

        else:
            if args.model == 'mobilefacenet_y2':
                self.model = Pruned_MobileFaceNet_y2(args.embedding_size).to(args.device)
                print('mobilefacenet_y2 mdoel generated')
            elif args.model == 'resnet50':
                self.model = fresnet50_v3().to(args.device)
                print('ResNet-50(公司) model generated')

        if args.pruned_checkpoint:
            state_dict = torch.load(args.pruned_checkpoint)
            self.model.load_state_dict(state_dict)
        else:
            print('重新训练一个模型')

        self.milestones = args.milestones
        self.loader, self.class_num = get_train_loader(args)

        self.writer = SummaryWriter(args.log_path)
        self.step = 0
        self.head = Arcface(embedding_size=args.embedding_size, classnum=self.class_num).to(args.device)
        if args.head_path:
            head_state_dict = torch.load(args.head_path)
            self.head.load_state_dict(head_state_dict)

        print('two model heads generated')

        paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

        self.optimizer = optim.SGD([
            {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
            {'params': paras_only_bn}
        ], lr=args.lr, momentum=args.momentum)

        print('optimizers generated')

        self.board_loss_every = len(self.loader) // 100
        self.evaluate_every = len(self.loader) // 10
        self.save_every = len(self.loader) // 5
        self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(args.test_root_path)

    def save_state(self, args, accuracy, to_save_folder=False, extra=None, model_only=False):
        import os
        if to_save_folder:
            save_path = args.save_path
        else:
            save_path = args.model_path
        torch.save(
            self.model.state_dict(), os.path.join(save_path, 'model_accuracy:{}_step:{}_{}.pth'.format(accuracy, self.step, extra)))

        if not model_only:
            torch.save(
            self.head.state_dict(), os.path.join(save_path, 'head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))
            # torch.save(
            #     self.optimizer.state_dict(), save_path /
            #     ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        self.model.load_state_dict(torch.load(save_path / 'model_{}'.format(fixed_str)))
        if not model_only:
            self.head.load_state_dict(torch.load(save_path / 'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)

    def evaluate(self, args, carray, issame, nrof_folds=5, tta=False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), args.embedding_size])
        # 得到每张图片的embeddings
        with torch.no_grad():
            while idx + args.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + args.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(args.device)) + self.model(fliped.to(args.device))
                    embeddings[idx:idx + args.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + args.batch_size] = self.model(batch.to(args.device)).cpu()
                idx += args.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(args.device)) + self.model(fliped.to(args.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(args.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    def train(self, args):
        self.model.train()
        running_loss = 0.
        best_acc_1 = 0.0
        best_acc_2 = 0.0
        best_acc_3 = 0.0
        for e in range(args.epoch):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()
            if e == self.milestones[2]:
                self.schedule_lr()
            for imgs, labels in tqdm(iter(self.loader)):
                imgs = imgs.to(args.device)
                labels = labels.to(args.device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                loss = nn.CrossEntropyLoss()(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.

                if self.step % self.evaluate_every == 0 and self.step != 0:
                    accuracy_1, best_threshold, roc_curve_tensor = self.evaluate(args, self.agedb_30,
                                                                                 self.agedb_30_issame)
                    self.board_val('agedb_30', accuracy_1, best_threshold, roc_curve_tensor)
                    accuracy_2, best_threshold, roc_curve_tensor = self.evaluate(args, self.lfw, self.lfw_issame)
                    self.board_val('lfw', accuracy_2, best_threshold, roc_curve_tensor)
                    accuracy_3, best_threshold, roc_curve_tensor = self.evaluate(args, self.cfp_fp, self.cfp_fp_issame)
                    self.board_val('cfp_fp', accuracy_3, best_threshold, roc_curve_tensor)
                    self.model.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    if accuracy_1 > best_acc_1:
                        self.save_state(args, accuracy_1, model_only=False, extra='best_acc_agedb_30')
                        best_acc_1 = accuracy_1
                    if accuracy_2 > best_acc_2:
                        self.save_state(args, accuracy_2, model_only=False, extra='best_acc_lfw')
                        best_acc_2 = accuracy_2
                    if accuracy_3 > best_acc_3:
                        self.save_state(args, accuracy_3, model_only=False, extra='best_acc_cfp_fp')
                        best_acc_3 = accuracy_3
                self.step += 1
            print(best_acc_1, best_acc_2, best_acc_3)

        # self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)