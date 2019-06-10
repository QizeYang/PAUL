import numpy as np
from PIL import Image
import random
import numbers
import math



class RandomCrop(object):
    """Crop the given PIL.Image to random size and aspect ratio.

    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, range=0.7, interpolation=Image.BILINEAR):
        self.range =  range

        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(100):
            area = img.size[0] * img.size[1]
            if isinstance(self.range, tuple):
                l, h = self.range
                ratio = random.uniform(l, h)
            else:
                assert isinstance(self.range, numbers.Number)
                ratio = self.range
            w = int(random.uniform(round(ratio*img.size[0]), img.size[0]))
            h = int(ratio* area/w)


            if w <= img.size[0] and h <= img.size[1]:
                left = random.randint(0, img.size[0] - w)
                upper = random.randint(0, img.size[1] - h)

                img = img.crop((left, upper, left + w, upper + h))
                assert (img.size == (w, h))

                return img

        # Fallback


        return img

