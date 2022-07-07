import os
import json
import numpy as np

import cv2


path = "/home/dc/workspace/datasets/OCR_rotate/crop1"

t_json = os.listdir(path)

jsons_names = []

for t_j in t_json:
    if t_j.endswith('.json'):
        jsons_names.append(t_j)


for json_name in jsons_names:
    json_path = os.path.join(path, json_name)
    with open(json_path, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        points = json_data['shapes'][0]['points']
        points = np.array(points, np.int32)
        imageHeight = json_data['imageHeight']
        imageWidth = json_data['imageWidth']
        image = np.zeros((imageHeight, imageWidth, 3))
        cv2.line(image, points[0], points[1], (255, 255, 255), 1)
        json_name = json_name.replace('.json', '_line.jpg')
        cv2.imwrite(os.path.join(path, json_name), image)
    # break


