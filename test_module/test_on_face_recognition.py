# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/11/1 下午5:04
import os
import torchvision.transforms.transforms as trans
from tqdm import tqdm
import torch
import numpy as np
from dataloader import Face_Recognition_Data
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import cv2
from src.data_loader import get_batch_test_data
import gc
import time


class TestOnFaceRecognition:
    def __init__(self, model, test_data_root, test_pair_list, dname):
        self.model = model
        self.test_data_root = test_data_root
        self.test_pair_list = test_pair_list
        self.image_list_path = test_pair_list.replace('pair','')
        self.image_pair_path = test_pair_list
        self.dname = dname
        self.embedding_result = {}
        self.id_index = []
        self.life_index = []

    def test(self, batch_size):
        name_space_path, imgs_list, actual_issame = get_val_dataset(self.test_data_root, self.test_pair_list, self.dname)
        return self.evaluate(self.model, name_space_path, imgs_list, actual_issame, batch_size)

    def evaluate(self, model, name_space_path, imgs_list, actual_issame, batch_size):
        features_dict = self.get_features_dict(name_space_path, model, batch_size)
        distance = np.zeros((int(len(imgs_list) / 2)))
        pair_num = int(len(imgs_list) / 2)
        print("-----------比对中----------")
        for j in tqdm(range(pair_num), desc='比对中'):
            lfw_img_name_1 = imgs_list[2 * j]
            lfw_img_name_2 = imgs_list[2 * j + 1]
            embed_1 = features_dict[lfw_img_name_1]
            embed_2 = features_dict[lfw_img_name_2]
            dist_all = np.linalg.norm(embed_1 - embed_2) * np.linalg.norm(embed_1 - embed_2)
            # dist_all = np.linalg.norm(embed_1 - embed_2)
            distance[j] = dist_all

        fpr, tpr, thresholds = roc_curve(actual_issame, -distance, pos_label=1)

        error = 10 ** -6
        fpr = np.around(fpr, decimals=7)
        index = np.argmin(abs(fpr - error))
        index_all = np.where(fpr == fpr[index])
        max_acc = np.max(tpr[index_all])
        # best_thresholds = np.max(abs(thresholds[index_all]))
        accuracy = np.around(max_acc, decimals=7)
        gen_plot(fpr, tpr, thresholds)
        return accuracy

    def get_features_dict(self, name_space_path, model, batch_size):
        model.eval()
        dict_name_features = {}
        with torch.no_grad():

            img_arr = [name_space_path[key][0] for key in name_space_path]
            img_key = [key for key in name_space_path]

            data = Face_Recognition_Data(img_arr, img_key, self.dname, batch_size)
            test_loader = data.test_loader
            print("----------特征提取中----------------")
            for img_info in tqdm(test_loader):
                imgs, keys = img_info
                features = model(imgs.cuda())
                for feature, key in zip(features, keys):
                    feature = feature.data.cpu().numpy()
                    dict_name_features[key] = feature

        return dict_name_features
    ############################################################################################################
    def test2(self, batch_size):
        self.extract_image_feature()
        accuarcy = self.make_comparation()
        return accuarcy

    def extract_image_feature(self):
        f = open(self.image_list_path, 'r')
        image_list = f.read().splitlines()
        f.close()

        print('Extract Feature')
        self.model.eval()
        with torch.no_grad():
            for elem_img in tqdm(image_list):
                image_path = '{}/{}'.format(self.test_data_root, elem_img)
                img = cv2.imread(image_path)
                tsr_img = self.cvimg_to_tensor(img)
                # emb_vec = self.model(tsr_img.to(self.device))
                # emb_vec = l2_norm(emb_vec)
                emb_vec = self.model.forward(tsr_img.cuda())
                self.embedding_result[elem_img] = emb_vec
        del image_list
        gc.collect()
        time.sleep(10)

    def make_comparation(self):
        f = open(self.image_pair_path, 'r')
        image_pairs = f.readlines()
        f.close()

        print('Make comparation')
        compare_result = []
        id_feature = self.embedding_result[self.id_index, :].float()
        life_feature = self.embedding_result[self.life_index, :].float()

        labels = []
        for line in image_pairs:
            label = float(line.split(' ')[2])
            labels.append(label)
        labels = np.array(labels)

        del image_pairs
        gc.collect()
        time.sleep(10)

        matrix = torch.mm(life_feature, id_feature.t())
        rows = matrix.shape[0] * matrix.shape[1]
        matrix = matrix.reshape(rows, 1)
        diff = 2 - 2 * matrix
        dist = np.array(diff)

        compare_result = np.zeros((len(labels), 2), dtype=np.float32)
        compare_result[:, 0] = dist[:, 0]
        compare_result[:, 1] = labels

        fpr, tpr, thresholds = roc_curve(compare_result[:,1], -compare_result[:,0], pos_label=1)

        error = 5e-6
        fpr = np.around(fpr, decimals=7)
        index = np.argmin(abs(fpr - error))
        index_all = np.where(fpr == fpr[index])
        max_acc = np.max(tpr[index_all])
        # best_thresholds = np.max(abs(thresholds[index_all]))
        accuracy = np.around(max_acc, decimals=7)
        gen_plot(fpr, tpr, thresholds)
        return accuracy

    def cvimg_to_tensor(self, img):
        img = img.astype(np.float32)/255
        tensor_img = torch.from_numpy(img.transpose((2,0,1)))
        tensor_img = torch.stack([tensor_img], 0)
        return tensor_img

    ############################################################################################################
    def test3(self, batch_size):
        self.get_image_id_life()
        self.extract_dataloader_feature(batch_size)
        accuarcy = self.make_comparation()
        return accuarcy

    def get_image_id_life(self):
        f = open(self.image_list_path, 'r')
        image_list = f.read().splitlines()
        f.close()

        for index, line in enumerate(image_list):
            if line.find("image_life") >= 0:
                self.life_index.append(index)
            if line.find("image_id") >= 0:
                self.id_index.append(index)

    def extract_dataloader_feature(self,batch_size):
        print("Extract Feature......")
        data_loader = get_batch_test_data(self.test_data_root, self.image_list_path,
                                          batch_size, 8)
        self.model.eval()
        with torch.no_grad():
            # for (img, img_prefix) in tqdm(data_loader):
            for (img, img_prefix) in tqdm(data_loader):
                emb_vec = self.model.forward(img.to(torch.device("cuda:0")))
                if len(self.embedding_result) == 0:
                    self.embedding_result = emb_vec.cpu()
                else:
                    self.embedding_result = torch.cat((self.embedding_result, emb_vec.cpu()))


def to_tensor(pic, data_source):

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            if data_source == 'company' or data_source == 'company_zkx':
                return img.float()
            elif data_source == 'lfw':
                return img.float().div(255)
        else:
            return img
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


class ToTensor:
    def __init__(self, data_source):
        self.data_source = data_source

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic, self.data_source)


def gen_plot(fpr, tpr, thresholds, title=None):
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    roc_info = []
    info_dict = {}
    for tolerance in [10 ** -7, 10 ** -6, 5.0 * 10 ** -6, 10 ** -5, 10 ** -4, 1.2 * 10 ** -4, 10 ** -3, 10 ** -2,
                      10 ** -1]:
        fpr = np.around(fpr, decimals=7)
        index = np.argmin(abs(fpr - tolerance))
        index_all = np.where(fpr == fpr[index])
        max_acc = np.max(tpr[index_all])
        threshold = np.max(abs(thresholds[index_all]))
        x, y = fpr[index], max_acc

        plt.plot(x, y, 'x')
        plt.text(x, y, "({:.9f}, {:.7f}) threshold={:.7f}".format(x, y, threshold))
        temp_info = 'fpr\t{}\tacc\t{}\tthreshold\t{}'.format(tolerance, round(max_acc, 5), threshold)
        roc_info.append(temp_info)
        info_dict[tolerance] = threshold

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - {}'.format(title))
    plt.savefig('roc.png', format='jpeg')
    plt.close()


def get_val_dataset(test_data_root, test_pair_list, dname):
    print("读取pair")
    pairs = read_pairs(test_pair_list)
    print("获取path")
    name_space_path, imgs_list, actual_issame = get_path_name(test_data_root, pairs, dname)
    return name_space_path, imgs_list, actual_issame


def read_pairs(test_pair_list):
    pairs = []
    with open(test_pair_list, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()
            pairs.append(pair)
    return pairs


def get_path_name(test_data_root, pairs, dname):
    name_space_path = {}
    imgs_list = []
    actual_issame = []
    for pair in pairs:
        img0_name = pair[0]
        img1_name = pair[1]
        issame = int(pair[2])

        imgs_list.append(img0_name)
        imgs_list.append(img1_name)
        actual_issame.append(issame)

        if dname == 'lfw':
            img0_path = os.path.join(test_data_root, img0_name[:-9], img0_name)
            img1_path = os.path.join(test_data_root, img1_name[:-9], img1_name)

        elif dname == 'company' or dname == 'company_zkx':
            img0_path = os.path.join(test_data_root, img0_name)
            img1_path = os.path.join(test_data_root, img1_name)

        name_space_path[img0_name] = []
        name_space_path[img1_name] = []
        name_space_path.setdefault(img0_name).append(img0_path)
        name_space_path.setdefault(img1_name).append(img1_path)

    return name_space_path, imgs_list, actual_issame
