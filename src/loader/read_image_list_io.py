#coding=utf-8
'''
DatasetFromList的作用
1.从list中读取图片的路径和标签
2.重载__getitem__属性，使得类的对象能够用[]操作符调用，返回图片和标签
3.这种做法将获取图片路径和加载图片集成到了一个类中完成，巧妙的做到了解耦
'''
import numpy as np
from torch.utils.data import Dataset
from .utility import read_image_list,read_image_triplet_list

class DatasetFromList(Dataset):
    """A generic data loader where the image list arrange in this way: ::

        class_x/xxx.ext 0
        class_x/xxy.ext 0
        class_x/xxz.ext 0

        class_y/123.ext 1
        class_y/nsdf3.ext 1
        class_y/asd932_.ext 1

    Args:
        root (string): Root directory path.
        image_list_path (string) : where to load image list
        loader (callable): A function to load a sample given its path.
        image_list_loader (callable) : A function to read image-label pair or image-image-prefix pair
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.<aimed to image>
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it. <aimed to label>

     Attributes:
        samples (list): List of (sample path, image_index) tuples
    """

    def __init__(self, root, image_list_path, loader ,
                 transform=None, target_transform=None, image_list_loader = read_image_list):
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

        Variable:
            self.samples (list): [image path, image label]

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        import os
        if not os.path.exists(path):
            print("Not exists..", path)

        sample = self.loader(path)
        assert isinstance(sample, np.ndarray) , path

        if self.transform is not None:
            sample = self.transform(sample)
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


class DatasetFromListTriplet(DatasetFromList):
    def __init__(self, root, image_list_path, loader ,
                 transform=None, target_transform=None, image_list_loader = read_image_triplet_list):
        super(DatasetFromListTriplet,self).__init__(root, image_list_path, loader, transform,
                                                    target_transform, image_list_loader)

    def __getitem__(self, index):
        """
        Args:
            index (tuple): includes (key, image_index)
        Variable:
            self.samples (dictionary): key(int) image_label, value(list) same label's image path

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        target = index[0]
        image_path = self.samples[index[0]][index[1]]
        sample = self.loader(image_path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def get_class_num(self):
        return len(self.samples)




