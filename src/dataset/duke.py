import utils.my_transforms as my_transforms
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

        class_ = i[0:4]
        if class_ not in classes:
            classes.append(class_)

    class_to_idx = {classes[i]: i for i in range(len(classes))}
    assert len(classes) == 702
    return classes, class_to_idx


def find_shared_classes(gallery, query):
    gallery_images = [d for d in os.listdir(gallery)]
    query_images = [d for d in os.listdir(query)]
    gallery_images.sort()
    query_images.sort()

    gallery_classes = []
    for i in gallery_images:
        class_ = i[0:4]
        if class_ not in gallery_classes:
            gallery_classes.append(class_)

    query_classes = []
    for i in query_images:
        class_ = i[0:4]
        if class_ not in query_classes:
            query_classes.append(class_)

    classes = list(set(gallery_classes)|set(query_classes))
    class_to_idx = {classes[i]: i for i in range(len(classes))}


    assert len(query_classes) == 702
    assert len(gallery_classes) == 702+408
    assert len(classes) == 702+408
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
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


class DUKE(data.Dataset):


    def __init__(self, root='../../data/DukeMTMC-reID', part='train', true_pair = False,
                 loader=default_loader, require_path=False, size=(384,128), pseudo_pair=0):

        self.root = root
        self.part = part
        self.loader = loader
        self.require_path = require_path
        self.true_pair = true_pair
        self.pseudo_pair = pseudo_pair
        self.subset = {'train': 'bounding_box_train',
                       'gallery': 'bounding_box_test',
                       'query': 'query',
                       }

        if part == 'gallery' or part == 'query':
            classes, class_to_idx = find_shared_classes(os.path.join(root, self.subset['gallery']),
                                                        os.path.join(root, self.subset['query']))
        else:
            classes, class_to_idx = find_classes(os.path.join(root, self.subset[part]))

        root = os.path.join(root, self.subset[part])
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                    "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        length = len(imgs)

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

        path, target, cam = self.imgs[index]


        img = self.loader(path)

        if self.true_pair:
            img = self.transform(img)
            random_index = list(range(self.len))
            random.shuffle(random_index)
            for i in random_index:
                tpath, ttarget, tcam = self.imgs[i]
                if ttarget ==  target:
                    timg = self.loader(tpath)
                    timg = self.transform(timg)
                    return img, target, path, cam, timg, tcam

        if self.pseudo_pair == 0:
            img = self.transform(img)
        else:
            img_list = [self.transform(img)]
            generator = transforms.Compose(
                [
                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                 transforms.RandomRotation(10),
                 my_transforms.RandomCrop(range=(0.7, 0.95)),
                 ])
            for i in range(self.pseudo_pair - 1):
                img_list.append(self.transform(generator(img)))
            img = torch.stack(img_list, dim=0)

        if self.require_path:
            return img, target, path, cam

        return img, target

    def __len__(self):
        return len(self.imgs)


