# -*- coding:utf-8 -*-
# author: LinX
# datetime: 2019/11/1 下午3:16
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torchvision.transforms import transforms
import cv2
from PIL import Image


def opencv_loader(path):
    img = cv2.imread(path, -1)
    return img


def read_image_list(root_path, image_list_path):
    f = open(image_list_path, 'r')
    data = f.read().splitlines()
    f.close()

    samples = []
    for line in data:
        sample_path = '{}/{}'.format(root_path, line.split(' ')[0])
        class_index = int(line.split(' ')[1])
        samples.append((sample_path, class_index))
    return samples


# ==========================================人脸识别数据加载====================================================
class Face_Recognition_Data:
    def __init__(self, img_arr, img_key, dname, batch_size):
        self.test_dataset = face_recognition_dataset(img_arr, img_key, dname)
        self.test_loader = DataLoader(self.test_dataset, batch_size, num_workers=4, pin_memory=True)


class face_recognition_dataset(Dataset):
    
    def __init__(self, img_arr, img_key, dname):
        self.dname = dname
        self.img_arr = img_arr
        self.img_key = img_key
        if dname == 'company' or dname == 'company_zkx':
            self.transform = ToTensor()
            self.loader = opencv_loader
        elif dname == 'lfw':
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __getitem__(self, item):
        img_path = self.img_arr[item]
        key = self.img_key[item]
        if self.dname == 'company' or self.dname == 'company_zkx':
            img = self.loader(img_path)
        elif self.dname == 'lfw':
            img = Image.open(img_path).convert('RGB')
        if self.transform:
            return self.transform(img), key
        return img, key

    def __len__(self):
        return len(self.img_arr)


# ========================================人脸分类数据加载======================================================
class Face_Classification_Data(Dataset):
    def __init__(self, img_root_path, image_list_path, batch_size):
        test_trans = ToTensor()
        self.test_dataset = face_classification_dataset(img_root_path, image_list_path, loader=opencv_loader, transform=test_trans,
                                                   image_list_loader=read_image_list)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)


class face_classification_dataset(Dataset):

    def __init__(self, root, image_list_path, loader,
                 transform=None, target_transform=None, image_list_loader=read_image_list):
        samples = image_list_loader(root, image_list_path)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in image_list: " + image_list_path + "\n"))

        self.root = root
        self.loader = loader
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if sample is None:
            print(path)
        assert sample is not None
        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except Exception as err:
                print('Error Occured: %s' % err, path)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def to_tensor(pic):

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
           return img.float()
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
        return img.float()
    else:
        return img


class ToTensor:
    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic)
