import os
import numpy as np
import json
from glob import glob
import cv2
from sklearn.model_selection import train_test_split
from shutil import copyfile
import argparse

obj_classes = []

# Labelme坐标到YOLO V5坐标的转换
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

# 样本转换
def convertToYolo5(fileList, output_dir, labelme_path):
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
    for json_file_ in fileList:
        # 1. 生成YOLO样本图片
        # 构建json图片文件的全路径名
        imagePath = labelme_path +'/'+ json_file_ + ".jpg"
        # 构建Yolo图片文件的全路径名
        yolo_image_file_path = yolo_images_dir + json_file_ + ".jpg"
        # copy样本图片
        copyfile (imagePath, yolo_image_file_path)
        
        # 2. 生成YOLO样本标签
        # 构建json标签文件的全路径名
        json_filename = labelme_path +'/'+ json_file_ + ".json"
        # 构建Yolo标签文件的全路径名
        yolo_label_file_path = yolo_labels_dir + json_file_ + ".txt"
        # 创建新的Yolo标签文件
        yolo_label_file = open(yolo_label_file_path, 'w')
        
        # 获取当前图片的Json标签文件
        json_obj = json.load(open(json_filename, "r", encoding="utf-8"))

        # 获取当前图片的长度、宽度信息
        height = json_obj['imageHeight']
        width  = json_obj['imageWidth']
        
        # 依次读取json文件中所有目标的shapes信息
        for shape in json_obj["shapes"]:
            # 获取shape中的物体分类信息
            label = shape["label"]
            if (label not in obj_classes):
                obj_classes.append(label)
            
            # 获取shape中的物体坐标信息
            if (shape["shape_type"] == 'rectangle'):
                points = np.array(shape["points"])
                xmin = min(points[:, 0]) if min(points[:, 0]) > 0 else 0
                xmax = max(points[:, 0]) if max(points[:, 0]) > 0 else 0
                ymin = min(points[:, 1]) if min(points[:, 1]) > 0 else 0
                ymax = max(points[:, 1]) if max(points[:, 1]) > 0 else 0
            
                # 对坐标信息进行合法性检查
                if xmax <= xmin:
                    pass
                elif ymax <= ymin:
                    pass
                else:
                    # Labelme坐标转换成YOLO V5坐标
                    bbox_labelme_float   = (float(xmin), float(xmax), float(ymin), float(ymax))
                    bbox_yolo_normalized = convert((width, height), bbox_labelme_float)
                    
                    # 把分类标签转换成分类id
                    class_id = obj_classes.index(label)
                    
                    # 生成YOLO V5的标签文件
                    yolo_label_file.write(str(class_id) + " " + " ".join([str(a) for a in bbox_yolo_normalized]) + '\n')
        yolo_label_file.close()
    
def check_output_directory(output = ""):
    # 创建保存输出图片的目录
    save_path = output + '/'
    is_exists = os.path.exists(save_path)
    
    if is_exists:
        print('Warning: path of %s already exist, please remove it firstly by manual' % save_path)
        #shutil.rmtree(save_path)  # 避免误删除已有的文件
        return ""
    
    #print('create output path %s' % save_path)
    os.makedirs(save_path)
    
    return save_path


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

def labelme2yolo(input = '', output = ''):

    outputdir_root = check_output_directory(output)
    if outputdir_root == "":
        print("No valid output directory, Do Nothing!")
        return -1
    
    labelme_path = input
    
    # 1.获取input目录中所有的json标签文件全路径名
    files = glob(labelme_path + "/*.json")
    
    # 2.获取所有标签文件的短文件名称
    files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]

    # print(files)
    # assert(0)
    
    # 3. 按比例随机切分数据集，获取训练集样本
    train_files, valid_test_files = train_test_split(files, test_size=0.3, random_state=55)
    
    # 4. 按比例随机切分数据集，获取验证集和测试集样本
    valid_files, test_files     = train_test_split(valid_test_files, test_size=0.3, random_state=55)

    # 5. 构建YOLO数据集目录
    train_path = outputdir_root+'/train'
    valid_path = outputdir_root+'/valid'
    test_path  = outputdir_root+'/test'
    
    # 6. 生成YOLO 训练、验证、测试数据集：图片+标签
    convertToYolo5(train_files, train_path, labelme_path)
    convertToYolo5(valid_files, valid_path, labelme_path)
    convertToYolo5(test_files,  test_path,  labelme_path)
    
    # 7. 创建YOLO数据集配置文件
    create_yolo_dataset_cfg(output, obj_classes)
    
    print("Classes:", obj_classes)
    print('Finished, output path =', outputdir_root)
    
    return 0
    
def parse_opt():
    # define argparse object
    parser = argparse.ArgumentParser()
    
    # add argument for command line
    parser.add_argument('--input',      type=str, help='The input Labelme directory')
    parser.add_argument('--output',     type=str, help='The output YOLO V5 directory')
    
    # parse arges from command line
    opt = parser.parse_args()
    print("input  =", opt.input)
    print("output =", opt.output)
    
    # return opt
    return opt

def main(opt):
    labelme2yolo(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)