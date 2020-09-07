from PIL import Image
import cv2

def read_image_list(root_path, image_list_path):
    f = open(image_list_path, 'r')
    # f = open(image_list_path, encoding='utf-8', mode='r')
    data = f.read().splitlines()
    f.close()

    samples = []
    for line in data:
        sample_path = '{}/{}'.format(root_path, line.split(' ')[0])
        class_index = int(line.split(' ')[1])
        samples.append((sample_path, class_index))
    return samples

def read_image_triplet_list(root_path, image_list_path):
    f = open(image_list_path, 'r')
    data = f.read().splitlines()
    f.close()

    label_num = int(data[-1].split(' ')[-1])+1
    samples = {}
    for i in range(label_num):
        samples[i] = []

    for line in data:
        sample_path = '{}/{}'.format(root_path, line.split(' ')[0])
        class_index = int(line.split(' ')[1])
        samples[class_index].append(sample_path)
    return samples


def read_image_list_test(root_path, image_list_path):
    f = open(image_list_path, 'r')
    data = f.read().splitlines()
    f.close()

    samples = []
    for line in data:
        sample_path = '{}/{}'.format(root_path, line.split(' ')[0])
        image_prefix = line.split(' ')[0]
        samples.append((sample_path, image_prefix))
    return samples


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
        #return img.convert('BGR')

def opencv_loader(path):
    img = cv2.imread(path)
    return img


# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


# def default_loader(path):
#     from torchvision import get_image_backend
#     if get_image_backend() == 'accimage':
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)