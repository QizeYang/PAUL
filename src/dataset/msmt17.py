import re
import torchvision.transforms as transforms
import random
import torch.utils.data as data
from PIL import Image
import os
import os.path
import time
import scipy.io as sio

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.mat', '.MAT'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



def make_dataset(root, config):

    with open(config, 'r') as f:
        lines = f.readlines()
    lines.sort()
    images = []
    classes = []

    for line in lines:
        filename, c = line.strip().split(' ')
        if c not in classes:
            classes.append(c)

    class_to_idx = {classes[i]: i for i in range(len(classes))}

    for line in lines:
        filename, c = line.strip().split(' ')
        if is_image_file(filename):
            path = os.path.join(root, filename)
            # print(filename)
            cam = int(filename[14:16])
            item = (path, class_to_idx[c], cam)
            images.append(item)


    return images, classes, class_to_idx


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            image = img.convert('RGB')
            img.close()
            return image


def default_loader(path):
    return pil_loader(path)


class MSMT17(data.Dataset):


    def __init__(self, root='../../data/MSMT17_V1', part='train', true_pair = False,
                 loader=default_loader, require_path=False, size=(384, 128)):

        self.root = root
        self.part = part
        self.loader = loader
        self.require_path = require_path
        self.true_pair = true_pair
        self.subset = {'train': 'list_train.txt',
                       'val': 'list_val.txt',
                       'gallery': 'list_gallery.txt',
                       'query': 'list_query.txt',
                       }
        if part in ['train','val']:
            imgs, classes, class_to_idx= make_dataset(os.path.join(root, 'train'), os.path.join(root, self.subset[part]))

        else:
            imgs, classes, class_to_idx = make_dataset(os.path.join(root, 'test'), os.path.join(root, self.subset[part]))



        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                     "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.class_num = len(classes)
        self.len = len(imgs)


        if part == 'train':
            self.transform = transforms.Compose([transforms.Resize(size),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])
        else:
            self.transform = transforms.Compose([transforms.Resize(size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])



    def __getitem__(self, index):

        path, target, cam = self.imgs[index]
        img = self.loader(path)
        img = self.transform(img)
        if self.true_pair:
            random_index = list(range(self.len))
            random.shuffle(random_index)
            for i in random_index:
                tpath, ttarget, tcam = self.imgs[i]
                if ttarget == target:
                    timg = self.loader(tpath)
                    timg = self.transform(timg)
                    return img, target, path, cam, timg, tcam
        if self.require_path:
            return img, target, path, cam

        return img, target


    def __len__(self):
        return len(self.imgs)





class MSMT17Extra(data.Dataset):


    def __init__(self, root='../../data/MSMT17_V1', part='train', true_pair = False,
                 loader=default_loader, require_path=False, size=(384, 128)):

        self.root = root
        self.part = part
        self.loader = loader
        self.require_path = require_path
        self.true_pair = true_pair
        self.subset = {'train': 'list_train.txt',
                       'val': 'list_val.txt',
                       'gallery': 'list_gallery.txt',
                       'query': 'list_query.txt',
                       }

        imgs, classes, class_to_idx = self._make_dataset(root)



        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                 "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.class_num = len(classes)
        self.len = len(imgs)



        self.transform = transforms.Compose([transforms.Resize(size),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                             ])



    def __getitem__(self, index):


        path, target, cam = self.imgs[index]
        img = self.loader(path)
        img = self.transform(img)
        if self.true_pair:
            random_index = list(range(self.len))
            random.shuffle(random_index)
            for i in random_index:
                tpath, ttarget, tcam = self.imgs[i]
                if ttarget == target:
                    timg = self.loader(tpath)
                    timg = self.transform(timg)
                    return img, target, path, cam, timg, tcam
        if self.require_path:
            return img, target, path, cam

        return img, target


    def __len__(self):
        return len(self.imgs)

    def _make_dataset(self, root):
        images = []
        classes = []
        for k, v in self.subset.items():
            if k in ['train','val']:
                image_list = os.path.join(root, v)
                prefix = 'train'
                current_root = os.path.join(root, prefix)

            else:
                image_list = os.path.join(root, v)
                prefix = 'test'
                current_root = os.path.join(root, prefix)

            with open(image_list, 'r') as f:
                lines = f.readlines()
            lines.sort()

            for line in lines:
                filename, c = line.strip().split(' ')
                c = c + prefix
                if c not in classes:
                    classes.append(c)

        class_to_idx = {classes[i]: i for i in range(len(classes))}

        for k, v in self.subset.items():

            if k in ['train','val']:
                image_list = os.path.join(root, v)
                prefix = 'train'
                current_root = os.path.join(root, prefix)

            else:
                image_list = os.path.join(root, v)
                prefix = 'test'
                current_root = os.path.join(root, prefix)

            with open(image_list, 'r') as f:
                lines = f.readlines()
            lines.sort()

            for line in lines:
                filename, c = line.strip().split(' ')
                c = c + prefix
                if is_image_file(filename):
                    path = os.path.join(current_root, filename)
                    # print(filename)
                    cam = int(filename[14:16])
                    item = (path, class_to_idx[c], cam)
                    images.append(item)

        return images, classes, class_to_idx

