import os
import functools
import random

from torch import rand

TRAIN_RATIO = 0.9
VALID_RATIO = 0.1

dataset_path = '../datasets/water_deep_voc/'
jpeg_path = os.path.join(dataset_path, 'JPEGImages')

dataset_img_list = os.listdir(jpeg_path)
dataset_list = list(map(lambda x: x[:-4], dataset_img_list))

dataset_size = len(dataset_list)
train_size = int(TRAIN_RATIO * dataset_size)
valid_size = dataset_size - train_size

train_set = dataset_list[:train_size]
valid_set = dataset_list[train_size:]

# 将训练集随机打乱
random.shuffle(dataset_list)

seg_path = os.path.join(dataset_path, 'ImageSets/Segmentation')

if not os.path.exists(seg_path):
    os.makedirs(seg_path)

with open(os.path.join(seg_path, 'train.txt'), 'w') as f:
    for data in train_set:
        f.write(data + '\n')

with open(os.path.join(seg_path, 'val.txt'), 'w') as f:
    for data in valid_set:
        f.write(data + '\n')

with open(os.path.join(seg_path, 'trainval.txt'), 'w') as f:
    for data in train_set + valid_set:
        f.write(data + '\n')