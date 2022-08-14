import json
import os
import random
from shutil import copyfile

import cv2
import numpy as np

outputdir_root = '../datasets/meter_yolo/'
paddle_path = '../datasets/meter_det/annotations/'

label_test = 'instance_test.json'
label_train = 'instance_train.json'

label_test_path = os.path.join(paddle_path, label_test)
label_train_path = os.path.join(paddle_path, label_train)

TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1

# xywh坐标到YOLO V5坐标的转换(标准化)
def convert(width, height, xywh):
    '''
    xywh = [x, y, w, h]
    '''
    x, y, w, h = xywh
    dw = 1. / width
    dh = 1. / height
    x += 0.5 * w
    y += 0.5 * h
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]
        

def load_paddle():
    with open(label_test_path) as test_file:
        test_dict = json.load(test_file)
        # print(test_dict['images'])

    with open(label_train_path) as train_file:
        train_dict = json.load(train_file)

    files = []

    for test_img in test_dict['images']:

        data_dict = dict()

        image_id = int(test_img['id'])
        file_name = str(test_img['file_name'])
        
        data_dict['image_id'] = image_id
        data_dict['file_name'] = file_name[:-4]
        data_dict['width'] = test_img['width']
        data_dict['height'] = test_img['height']
        data_dict['bbox'] = []
        data_dict['category_id'] = []

        files.append(data_dict)

        # print(f'Image {image_id} loaded.')

    test_size = len(files)

    for train_img in train_dict['images']:

        data_dict = dict()

        image_id = int(train_img['id']) + test_size - 1
        file_name = str(train_img['file_name'])
        
        data_dict['image_id'] = image_id
        data_dict['file_name'] = file_name[:-4]
        data_dict['width'] = test_img['width']
        data_dict['height'] = test_img['height']
        data_dict['bbox'] = []
        data_dict['category_id'] = []

        files.append(data_dict)

        # print(f'Image {image_id} loaded.')

    # print(f'\n\n{len(files)} images loaded, start bbox loading.')

    for test_anno in test_dict['annotations']:
        image_id = int(test_anno['image_id'])
        width, height = files[image_id]['width'], files[image_id]['height']
        bbox_id = int(test_anno['id'])
        bbox = convert(width, height, test_anno['bbox'])
        category_id = str(test_anno['category_id'])

        files[image_id]['bbox'].append(bbox)
        files[image_id]['category_id'].append(category_id)
        # print(f'bbox {bbox_id} of Image {image_id} loaded.')
        # print(files[image_id])
        
    for train_anno in train_dict['annotations']:
        image_id = int(train_anno['image_id']) + test_size - 1
        width, height = files[image_id]['width'], files[image_id]['height']
        bbox_id = int(train_anno['id'])
        bbox = convert(width, height, train_anno['bbox'])
        category_id = str(train_anno['category_id'])

        files[image_id]['bbox'].append(bbox)
        files[image_id]['category_id'].append(category_id)

        # print(f'bbox {bbox_id} of Image {image_id} loaded.')
        # print(files[image_id])
    return files

def convert_to_yolo(files, output_dir, paddle_path):
    # 创建指定样本的父目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建指定样本的images和labels子目录
    yolo_images_dir = '{}/images/'.format(output_dir)
    yolo_labels_dir = '{}/labels/'.format(output_dir)
    
    if not os.path.exists(yolo_images_dir):
        os.makedirs(yolo_images_dir)
    if not os.path.exists(yolo_labels_dir):
        os.makedirs(yolo_labels_dir)
    
    # 一个样本图片一个样本图片地转换
    for file in files:
        # 1. 生成YOLO样本图片
        # 构建json图片文件的全路径名
        img_name = file['file_name']
        img_path = os.path.join(paddle_path, img_name + ".jpg")
        # 构建Yolo图片文件的全路径名
        yolo_image_file_path = os.path.join(yolo_images_dir, img_name + ".jpg")
        # copy样本图片
        copyfile(img_path, yolo_image_file_path)
        
        # 2. 生成YOLO样本标签
        # 构建json标签文件的全路径名
        # json_filename = paddle_path +'/'+ file + ".json"
        # 构建Yolo标签文件的全路径名
        yolo_label_file_path = os.path.join(yolo_labels_dir, img_name + ".txt")
        # 创建新的Yolo标签文件
        yolo_label_file = open(yolo_label_file_path, 'w')

        # 获取当前图片的长度、宽度信息
        height = file['height']
        width  = file['width']
        
        # 依次读取json文件中所有目标的shapes信息
        for bbox, category_id in zip(file['bbox'], file['category_id']):            
            # 把分类标签转换成分类id
            class_id = int(category_id) - 1
            
            # 生成YOLO V5的标签文件
            yolo_label_file.write(str(class_id) + " " + " ".join([str(a) for a in bbox]) + '\n')
        yolo_label_file.close()

def create_yolo_dataset_cfg(output_dir='', label_class = []):
    # 创建文件
    data_cfg_file = open(output_dir + '/data.yaml', 'w')
    
    # 创建文件内容
    data_cfg_file.write('train:  ../train/images\n')
    data_cfg_file.write("val:    ../valid/images\n")
    data_cfg_file.write("test:   ../test/images\n")
    data_cfg_file.write("\n")
    data_cfg_file.write("# Classes\n")
    data_cfg_file.write("nc: %s\n" %len(label_class))
    data_cfg_file.write('names: ')
    i = 0
    for label in label_class:
        if (i == 0):
            data_cfg_file.write("[")
        else:
            data_cfg_file.write(", ")
            if  (i % 10 == 0):
                data_cfg_file.write("\n        ")
        i += 1
        data_cfg_file.write("'" + label + "'")
    data_cfg_file.write(']  # class names')
    data_cfg_file.close()
    #关闭文件

if __name__ == '__main__':
    files = load_paddle()

    random.shuffle(files)

    dataset_size = len(files)

    train_files = files[:int(dataset_size * TRAIN_RATIO)]
    valid_files = files[int(dataset_size * TRAIN_RATIO):int(dataset_size * (TRAIN_RATIO + VALID_RATIO))]
    test_files = files[int(dataset_size * (TRAIN_RATIO + VALID_RATIO)):]

    train_path = outputdir_root+'/train'
    valid_path = outputdir_root+'/valid'
    test_path  = outputdir_root+'/test'
    
    # 6. 生成YOLO 训练、验证、测试数据集：图片+标签
    convert_to_yolo(train_files, train_path, paddle_path)
    convert_to_yolo(valid_files, valid_path, paddle_path)
    convert_to_yolo(test_files,  test_path, paddle_path)
    # print(files)
    # for file in files:
        # print(file['image_id'])
    obj_classes = ['meter']
    create_yolo_dataset_cfg(outputdir_root, obj_classes)
    
    print("Classes:", obj_classes)
    print('Finished, output path =', outputdir_root)


