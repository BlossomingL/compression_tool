'''
 Data load from image list
 1. with no Normalize
 2. with opencv load image
 3. image is not divide 255
 @19-10-21 mod by ljt: add data augmentation
 @19-11-29 mod by zkx 调整数据预处理方式到dict结构进行调用
'''

import os
from src.loader import transforms_V2 as trans
from src.loader.autoaugment import ImageNetPolicy
from torch.utils.data import DataLoader
from src.loader.utility import opencv_loader, read_image_list_test
from src.loader.read_image_list_io import DatasetFromList, DatasetFromListTriplet
import torch

pwd_path = os.path.dirname(__file__)
from src.dataset import TrainSet,TestSet


class DefineTrans:
    def __init__(self, input_size):
        self.image_preprocess = {
            'D1':trans.Compose([
                    trans.ToPILImage(),
                    trans.RandomResizedCrop(input_size, scale=(0.99, 1.01)),
                    trans.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                    trans.RandomHorizontalFlip(),
                    trans.ToTensor(),
                ]),
            'D2':trans.Compose([
                    trans.ToPILImage(),
                    trans.RandomResizedCrop(input_size, scale=(0.9, 1.1)),
                    trans.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                    trans.RandomHorizontalFlip(),
                    trans.ToTensor(),
                ]),
            'D3':  trans.Compose([
                    trans.ToPILImage(),
                    trans.RandomResizedCrop(input_size, scale=(0.9, 1.1)),
                    trans.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                    trans.RandomRotation(5),
                    trans.RandomHorizontalFlip(),
                    trans.ToTensor(),
                ]),
            'D9':trans.Compose([
                    trans.ToPILImage(),
                    trans.RandomHorizontalFlip(),
                    ImageNetPolicy(),
                    trans.ToTensor(),
                ]),
            'None':trans.Compose([
                    trans.ToTensor(),
                ])

            # others
            # trans.RandomPerspective(),
            # trans.RandomErasing(),
            # trans.RandomRotation(5),
            # trans.RandomApply([trans.RandomResizedCrop((width, height), scale=(0.99, 1.01))]),
            # trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        }

# 用于训练集的数据加载
def get_train_dataset(img_root_path, image_list_path, data_aug):
    height = int(img_root_path.split('_')[-1].split('x')[0])
    width = int(img_root_path.split('_')[-1].split('x')[1])

    # @2019-11-29 zkx get input image process method from dict
    input_process = DefineTrans((height, width))
    if data_aug is None:
        train_transform = input_process.image_preprocess['None']
    else:
        train_transform = input_process.image_preprocess[data_aug]

    ds = DatasetFromList(img_root_path, image_list_path, opencv_loader, train_transform, None)
    class_num = ds[-1][1] + 1
    return ds, class_num


def get_train_list_loader(conf):
    train_set = conf.data_mode
    img_root_path = TrainSet[train_set]['root_path']
    img_label_list = TrainSet[train_set]['label_list']

    print("img_label_list")
    print(img_label_list)

    patch_info = conf.patch_info
    root_path = '{}/{}'.format(img_root_path, patch_info)
    celeb_ds, celeb_class_num = get_train_dataset(root_path, img_label_list, conf.data_aug)
    print("images:")
    print(len(celeb_ds))
    ds = celeb_ds
    class_num = celeb_class_num
    if conf.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(ds)
    else:
        train_sampler = None
    # @2020-02-25 ljt 加入drop_last，解决bn在最后一个step报错的问题
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=(train_sampler is None), pin_memory=conf.pin_memory,
                        num_workers=conf.num_workers, sampler=train_sampler,drop_last=True)
    return loader, class_num, train_sampler


# 用于测试集的数据加载
def get_test_dataset(img_root_path, image_list_path):

    # @2019-11-29 zkx get input image process method from dict
    input_process = DefineTrans((1, 1))
    test_transform = input_process.image_preprocess["None"]
    ds = DatasetFromList(img_root_path, image_list_path, opencv_loader, test_transform, None, read_image_list_test)
    return ds

def get_batch_test_data(image_root_path, image_list_path, batch_size, num_workers):
    ds = get_test_dataset(image_root_path, image_list_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return loader

