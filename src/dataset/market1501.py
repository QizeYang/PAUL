import torch
import re
import torchvision.transforms as transforms
import random
import torch.utils.data as data
from PIL import Image
import os
import os.path
import time
import scipy.io as sio
import utils.my_transforms as my_transforms
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.mat', '.MAT'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    images = [d for d in os.listdir(dir)]
    images.sort()
    classes = []
    for i in images:
        if "-1" in i:
            continue
        class_ = i[0:4]

        if class_ == "0000":
            continue
        if class_ not in classes:
            classes.append(class_)
    classes.append("0000")
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx



def make_dataset(dir, class_to_idx, cam_filter=None):
    images = []
    dir = os.path.expanduser(dir)

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if "-1" in fname:
                continue
            target = fname[0:4]
            if is_image_file(fname):
                path = os.path.join(root, fname)
                cam = int(fname[6])
                if cam_filter is not None:
                    if cam_filter != cam:
                        continue
                item = (path, class_to_idx[target], cam)
                images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            image = img.convert('RGB')
            img.close()
            return image


def default_loader(path):
    return pil_loader(path)


class MARKET(data.Dataset):


    def __init__(self, root='../../data/Market', part='train', true_pair=False,
                 loader=default_loader, require_path=False, size=(384, 128), cam_filter=None, pseudo_pair=0):

        self.root = root
        self.part = part
        self.loader = loader
        self.require_path = require_path
        self.pseudo_pair = pseudo_pair
        self.size = size
        self.true_pair = true_pair

        self.subset = {'train': 'bounding_box_train',
                       'gallery': 'bounding_box_test',
                       'query': 'query',
                       }

        root = os.path.join(root, self.subset[part])
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx, cam_filter)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        if part == 'train':

            self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.Resize(size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])


        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.len = len(imgs)
        self.class_num = len(classes)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target, cam = self.imgs[index]


        img = self.loader(path)

        if self.true_pair:
            img = self.transform(img)
            random_index = list(range(self.len))
            random.shuffle(random_index)
            for i in random_index:
                tpath, ttarget, tcam = self.imgs[i]
                if ttarget == target:
                    timg = self.loader(tpath)
                    timg = self.transform(timg)
                    return img, target, path, cam, timg, tcam

        if self.pseudo_pair == 0:
            img = self.transform(img)
        else:
            img_list = [self.transform(img)]
            generator = transforms.Compose(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 transforms.RandomRotation(10),
                 my_transforms.RandomCrop(range=(0.70,0.95)),
                 ])
            for i in range(self.pseudo_pair-1):
                img_list.append(self.transform(generator(img)))
            img = torch.stack(img_list, dim=0)


        if self.require_path:
            return img, target, path, cam

        return img, target

    def __len__(self):
        return len(self.imgs)



