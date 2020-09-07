import torch
from PIL import Image
from commom_utils.utils import hflip_batch, gen_plot, l2_norm
from torchvision import transforms as trans
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import sklearn
import bcolz
import os


class TestWithInsightFace:
    def __init__(self, model):
        self.model = model

    def get_val_pair(self, path, name):
        carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
        issame = np.load(os.path.join(path, '{}_list.npy'.format(name)))
        return carray, issame

    def get_val_data(self, data_path):
        agedb_30, agedb_30_issame = self.get_val_pair(data_path, 'agedb_30')  # 12000图片对，6000个相对应的标签
        cfp_fp, cfp_fp_issame = self.get_val_pair(data_path, 'cfp_fp')
        lfw, lfw_issame = self.get_val_pair(data_path, 'lfw')
        return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame

    def test(self, carray, issame, batch_size, device, nrof_folds=5, tta=False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), 512])
        # 得到每张图片的embeddings
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(device)) + self.model(fliped.to(device))
                    embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + batch_size] = self.model(batch.to(device)).cpu()
                idx += batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(device)) + self.model(fliped.to(device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(device)).cpu()
        tpr, fpr, accuracy, best_thresholds = self.evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean()

    def calculate_accuracy(self, threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

        tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
        acc = float(tp + tn) / dist.size
        return tpr, fpr, acc

    def calculate_roc(self, thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
        assert (embeddings1.shape[0] == embeddings2.shape[0])
        assert (embeddings1.shape[1] == embeddings2.shape[1])
        nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
        nrof_thresholds = len(thresholds)
        k_fold = KFold(n_splits=nrof_folds, shuffle=False)

        tprs = np.zeros((nrof_folds, nrof_thresholds))
        fprs = np.zeros((nrof_folds, nrof_thresholds))
        accuracy = np.zeros((nrof_folds))
        best_thresholds = np.zeros((nrof_folds))
        indices = np.arange(nrof_pairs)
        # print('pca', pca)

        if pca == 0:
            diff = np.subtract(embeddings1, embeddings2)  # 特征相减
            dist = np.sum(np.square(diff), 1)

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            # print('train_set', train_set)
            # print('test_set', test_set)
            if pca > 0:
                print('doing pca on', fold_idx)
                embed1_train = embeddings1[train_set]
                embed2_train = embeddings2[train_set]
                _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
                # print(_embed_train.shape)
                pca_model = PCA(n_components=pca)
                pca_model.fit(_embed_train)
                embed1 = pca_model.transform(embeddings1)
                embed2 = pca_model.transform(embeddings2)
                embed1 = sklearn.preprocessing.normalize(embed1)
                embed2 = sklearn.preprocessing.normalize(embed2)
                # print(embed1.shape, embed2.shape)
                diff = np.subtract(embed1, embed2)
                dist = np.sum(np.square(diff), 1)

            # Find the best threshold for the fold
            acc_train = np.zeros((nrof_thresholds))
            for threshold_idx, threshold in enumerate(thresholds):
                _, _, acc_train[threshold_idx] = self.calculate_accuracy(threshold, dist[train_set],
                                                                    actual_issame[train_set])
            best_threshold_index = np.argmax(acc_train)
            #         print('best_threshold_index', best_threshold_index, acc_train[best_threshold_index])
            best_thresholds[fold_idx] = thresholds[best_threshold_index]
            for threshold_idx, threshold in enumerate(thresholds):
                tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = self.calculate_accuracy(threshold,
                                                                                                     dist[test_set],
                                                                                                     actual_issame[
                                                                                                         test_set])
            _, _, accuracy[fold_idx] = self.calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                          actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        return tpr, fpr, accuracy, best_thresholds

    def evaluate(self, embeddings, actual_issame, nrof_folds=10, pca=0):
        # Calculate evaluation metrics
        thresholds = np.arange(0, 4, 0.01)
        embeddings1 = embeddings[0::2]
        embeddings2 = embeddings[1::2]
        tpr, fpr, accuracy, best_thresholds = self.calculate_roc(thresholds, embeddings1, embeddings2,
                                                            np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca)
        # thresholds = np.arange(0, 4, 0.001)
        # val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
        #                                   np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
        # return tpr, fpr, accuracy, best_thresholds, val, val_std, far
        return tpr, fpr, accuracy, best_thresholds

