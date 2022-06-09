import cv2
import numpy as np
import torch


def get_numpy_img_from_image(image, shape):
    img = image
    ori_frame = img.copy()
    img = cv2.resize(img, shape)
    return ori_frame, img


if __name__ == "__main__":
    plate = cv2.imread("test_images/1.jpg")
    model = torch.jit.load('checkpoints/mask.torchscript')
    ori_image, plate = get_numpy_img_from_image(plate, shape=(224, 56))
    plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
    plate = np.array(plate)
    plate = plate.transpose(2, 0, 1) / 255.0
    plate = torch.tensor(plate).float().unsqueeze(0)
    output = model(plate)
    print(output)
    print(output.argmax(dim=1))

