from cProfile import label
import os
import random

import torch
import torchvision

from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader


class Basic_Dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.images_list = os.listdir(data_path)
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image_name = self.images_list[index]
        image_path = os.path.join(self.data_path, image_name)
        image = Image.open(image_path)
        if random.random() > 0.5:
            size = image.size()
            mask = Image.new("RGB", size, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            image.paste(mask, (0, 0, size[0], size[1]))
            image = self.transform(image)
            label = 1
            return image, label
        else:
            image = self.transform(image)
            label = 0
            return image, label


