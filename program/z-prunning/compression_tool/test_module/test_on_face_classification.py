# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/11/1 下午6:14
import cv2
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
from dataloader import Face_Classification_Data
from tqdm import tqdm


def opencv_loader(path):
    img = cv2.imread(path, -1)
    return img


class TestOnFaceClassification:

    def __init__(self, model, root_path, list_label, device=0, t=1e-3):
        self.model = model
        self.root_path = root_path
        self.list_label = list_label
        self.image_root_path = root_path
        self.list_label = list_label
        self.device = device
        self.t = t

    def test(self, batch_size):
        self.model.eval()
        self.model.to(self.device)
        data_loader = Face_Classification_Data(self.root_path, self.list_label, batch_size).test_loader
        file = open(self.list_label, 'r')
        liens = file.readlines()
        file.close()

        label_list = []
        score_list = []
        with torch.no_grad():
            for img, label in tqdm(data_loader):
                input = img.to(self.device)
                out = self.model(input)
                out = F.softmax(out, dim=1)

                for idx in range(len(out)):

                    label_list.append(label.cpu().numpy()[idx])
                    score_list.append(out[idx].cpu().numpy()[1])

        fpr, tpr, thresholds = roc_curve(np.array(label_list), np.array(score_list), pos_label=1)
        fpr = np.around(fpr, decimals=7)
        index = np.argmin(abs(fpr - self.t))
        index_all = np.where(fpr == fpr[index])
        max_acc = np.max(tpr[index_all])

        return max_acc

