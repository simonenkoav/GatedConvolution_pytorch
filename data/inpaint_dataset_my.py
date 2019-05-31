import torch
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image, ImageOps
from .base_dataset import BaseDataset, NoriBaseDataset
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import random


def find_coeffs(source_coords, target_coords):
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]])
    A = np.matrix(matrix, dtype=np.float)
    B = np.array(source_coords).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def random_crop(image):
    new_w, new_h = int(round(image.width * random.uniform(0.9, 1.0))), \
                   int(round(image.height * random.uniform(0.9, 1.0)))
    start_x, start_y = (image.width - new_w)/2, (image.height - new_w)/2
    return image.crop((start_x, start_y, start_x + new_w, start_y + new_h)).resize(image.size)


def transform_image(image):
    width, height = image.size
    m = random.uniform(-0.3, 0.3)
    xshift = m * width
    new_width = width + int(round(xshift))
    coeffs = find_coeffs(
        [(0, 0), (image.size[0], 0), (image.size[0], image.size[0]), (0, image.size[0])],
        [(0, 0), (image.size[0], 0), (new_width, height), (xshift, height)])
    return random_crop(image.transform(image.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC))


class InpaintDatasetMy(BaseDataset):
    """
    Dataset for Inpainting task
    Params:
        img_flist_path(str): The file which contains img file path list (e.g. test.flist)
        mask_flist_paths_dict(dict): The dict contain the files which contains the pkl or xml file path for
                                generate mask. And the key represent the mask type (e.g. {"bbox":"bbox_flist.txt", "seg":..., "random":None})
        resize_shape(tuple): The shape of the final image (default:(256,256))
        transforms_oprs(list) : Determine which transformation used on the imgae (default:['random_crop', 'to_tensor'])
        random_bbox_shape(tuple): if use random bbox mask, it define the shape of the mask (default:(32,32))
        random_bbox_margin(tuple): if use random bbox, it define the margin of the bbox which means the distance between the mask and the margin of the image
                                    (default:(64,64))
    Return:
        img, *mask
    """
    def __init__(self, flist_path, base_path, resize_shape=(256, 256), transforms_oprs=['to_tensor']):

        self.data = []

        for line in open(base_path + flist_path):
            iname, bbox, ename = line.rstrip().split('\t')
            self.data.append((iname, list(map(int, bbox.split())), ename))
        self.base_path = base_path

        self.resize_shape = resize_shape
        self.transform_initialize(resize_shape, transforms_oprs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # create the paths for images and masks

        iname, bbox, mname = self.data[index]
        image = self.read_img(self.base_path + iname)
        edge_image = Image.open(self.base_path + mname)
        edge_image = transform_image(edge_image)

        mask = self.process_mask(image.size, bbox)
        drawing = self.process_drawing(edge_image, image.size, bbox)
        image, mask, drawing = image.resize(self.resize_shape), mask.resize(self.resize_shape), \
                               drawing.resize(self.resize_shape)
        image, mask, drawing = self.transforms_fun(image), self.transforms_fun(mask), self.transforms_fun(drawing)

        return image*255, mask*255, drawing*255

    def read_img(self, path):
        img = Image.open(path).convert("RGB")
        return img

    @staticmethod
    def process_mask(img_size, bbox):
        mask = np.zeros(img_size)
        mask[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1] = 1
        return Image.fromarray(mask)

    @staticmethod
    def process_drawing(mask, img_size, bbox):
        drawing = mask.resize(img_size)
        drawing = ImageOps.invert(drawing)
        drawing = np.array(drawing)
        drawing[drawing > 105] = 255
        drawing[drawing != 255] = 0
        for i in range(drawing.shape[0]):
            for k in range(drawing.shape[1]):
                if not (bbox[1] <= i <= bbox[3] and bbox[0] <= k <= bbox[2]):
                    drawing[i][k] = 0
        return Image.fromarray(drawing)
