from cProfile import label
import os
import random
from this import s
import numpy as np
import math

import torch
import torchvision

from PIL import Image
import cv2
from torch.utils.data import Dataset

def img_float32(img):
    return img.copy() if img.dtype != 'uint8' else (img/255.).astype('float32')

def over(fgimg, bgimg):
    fgimg, bgimg = img_float32(fgimg),img_float32(bgimg)
    (fb,fg,fr,fa),(bb,bg,br,ba) = cv2.split(fgimg),cv2.split(bgimg)
    #分别得到bgr图和透明图
    # [[[b,g,r], [b,g,r], [b,g,r]]
    #  [[b,g,r], [b,g,r], [b,g,r]]]
    color_fg, color_bg = cv2.merge((fb,fg,fr)), cv2.merge((bb,bg,br))
 
    # [[a,a,a,a]     [[[a], [a], [a], [a]]
    #  [a,a,a,a]] ->  [[a], [a], [a], [a]]]
    alpha_fg, alpha_bg = np.expand_dims(fa, axis=-1), np.expand_dims(ba, axis=-1)
    
    color_fg[fa==0]=[0,0,0]
    color_bg[ba==0]=[0,0,0]
    
    #outA = srcA + dstA(1 - srcA)
    a = fa + ba * (1-fa)
 
    #outRGB = (srcRGBsrcA + dstRGBdstA(1 - srcA)) / outA
    color_over = (color_fg * alpha_fg + color_bg * alpha_bg * (1-alpha_fg)) / np.expand_dims(a, axis=-1)
    color_over = np.clip(color_over,0,1)
    
    #out A = 0 -> outRGB = 0
    color_over[a==0] = [0,0,0]
    
    result_float32 = np.append(color_over, np.expand_dims(a, axis=-1), axis = -1)
    return (result_float32*255).astype('uint8')

def overlay_with_transparency(bgimg, fgimg, xmin = 0, ymin = 0,trans_percent = 1):
    '''
    bgimg: a 4 channel image, use as background
    fgimg: a 4 channel image, use as foreground
    xmin, ymin: a corrdinate in bgimg. from where the fgimg will be put
    trans_percent: transparency of fgimg. [0.0,1.0]
    '''
    #we assume all the input image has 4 channels
    assert(bgimg.shape[-1] == 4 and fgimg.shape[-1] == 4)
    fgimg = fgimg.copy()
    # 获取叠加位置
    roi = bgimg[ymin:ymin+fgimg.shape[0], xmin:xmin+fgimg.shape[1]].copy()
    
    b,g,r,a = cv2.split(fgimg)
    
    fgimg = cv2.merge((b,g,r,(a*trans_percent).astype(fgimg.dtype)))
    roi_over = over(fgimg,roi)
    
    result = bgimg.copy()
    result[ymin:ymin+fgimg.shape[0], xmin:xmin+fgimg.shape[1]] = roi_over
    return result

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
            mask_size = (size[0] // random.randint(1,2), size[1] // random.randint(1,2))
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


class Mask_Dataset(Dataset):
    def __init__(self, data_path, mask_path):
        self.data_path = data_path
        self.mask_path = mask_path
        self.images_list = os.listdir(data_path)
        self.mask_list = os.listdir(mask_path)
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((56, 224)),
                ]
            )

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image_name = self.images_list[index]
        image_path = os.path.join(self.data_path, image_name)
        
        if random.random() > 0.5:
            image = cv2.imread(image_path)
            image = self.overlap_target(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = self.transform(image)
            label = 1
            return image, label
        else:
            image = Image.open(image_path)
            image = self.transform(image)
            label = 0
            return image, label

    def overlap_target(self, img):
        img = cv2.resize(img, (224, 56))
        mask_index = random.randint(1, len(self.mask_list) - 1)
        mask_path = os.path.join(self.mask_path, f"{mask_index}.png")
        dirty_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        dw, dh = math.ceil(224 * random.randint(80, 100)/100.0), math.ceil(56 * random.randint(80, 100)/100.0)
        dirty_img = cv2.resize(dirty_img, (dw, dh))
        M = cv2.getRotationMatrix2D((dirty_img.shape[1] / 2, dirty_img.shape[0] / 2), random.randint(0, 360), 1)
        dirty_img = cv2.warpAffine(dirty_img, M, (dirty_img.shape[1], dirty_img.shape[0]))
        px, py = random.randint(0, 224-dw), random.randint(0, 56-dh)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img = overlay_with_transparency(img, dirty_img, px, py)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


class Line_Dataset(Dataset):
    def __init__(self, data_path='/home/dc/workspace/datasets/OCR_rotate/crop1'):
        self.data_path = data_path
        self.images = []
        self.labels = []
        files_list = os.listdir(data_path)
        for file in files_list:                
            if file.endswith('_line.jpg'):
                self.labels.append(os.path.join(data_path, file))
            elif file.endswith('.jpg'):
                self.images.append(os.path.join(data_path, file))
        self.transform_image = torchvision.transforms.Compose(
            [
                torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((56, 224)),
                ]
            )
        self.transform_label = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((56, 224)),
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]
        label_path = self.labels[index]
        image = Image.open(image_path)
        label = Image.open(label_path).convert("L")
        image = self.transform_image(image)
        label = self.transform_label(label)
        label_max_index = torch.argmax(label, dim=1)
        label = label_max_index / label.shape[1]

        return image, label








if __name__ == "__main__":
    data = Line_Dataset()
    
    image, label = data[0]
    print(label.max())
    image = torchvision.transforms.ToPILImage()(image)
    image.save("tmp.jpg")
    print(label)

    # data = torch.rand((1, 5, 8))
    # print(data)
    # data_arg = torch.argmax(data, dim=1)
    # print(data_arg)
