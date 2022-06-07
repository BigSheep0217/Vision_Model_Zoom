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
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((56, 224)),
                ]
            )

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image_name = self.images_list[index]
        image_path = os.path.join(self.data_path, image_name)
        image = Image.open(image_path)
        if random.random() > 0.5:
            size = image.size
            mask_size = (size[0] // random.randint(2,4), size[1] // random.randint(1,2))
            mask = Image.new("RGB", mask_size, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            paste_area = (random.randint(0, size[0] - mask_size[0]), random.randint(0, size[1] - mask_size[1]))
            image.paste(mask, (paste_area[0], paste_area[1], paste_area[0] + mask_size[0], paste_area[1] + mask_size[1]))
            image = self.transform(image)
            label = 1
            return image, label
        else:
            image = self.transform(image)
            label = 0
            return image, label


if __name__ == "__main__":
    data = Basic_Dataset(r"D:\Samples\done\sandun\part12\crop\plate_blue")
    
    image, label = data[0]
    image = torchvision.transforms.ToPILImage()(image)
    image.show()
    print(label)
