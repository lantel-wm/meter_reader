import os
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def gauss_noise(img, seg, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    img = np.array(img, dtype=float) / 255
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out = img + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    dst = Image.fromarray(np.uint8(out))

    return dst, seg

def random_color(img, seg):
    '''
    色彩抖动
    '''
    saturation = random.randint(0,1) # 饱和度
    brightness = random.randint(0,1) # 亮度
    contrast = random.randint(0,1) # 对比度
    sharpness = random.randint(0,1) # 锐度

    if random.random() < saturation:
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        img = ImageEnhance.Color(img).enhance(random_factor)  # 调整图像的饱和度

    if random.random() < brightness:
        random_factor = np.random.randint(10, 14) / 10.  # 随机因子
        img = ImageEnhance.Brightness(img).enhance(random_factor)  # 调整图像的亮度

    if random.random() < contrast:
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        img = ImageEnhance.Contrast(img).enhance(random_factor)  # 调整图像对比度

    if random.random() < sharpness:
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        ImageEnhance.Sharpness(img).enhance(random_factor)  # 调整图像锐度
        
    return img, seg

def rotate(img, seg, angle):
    '''
        旋转
        angle: 角度
    '''
    dst_img = img.rotate(angle)
    dst_seg = seg.rotate(angle)

    return dst_img, dst_seg

def data_enhance(img, seg):
    '''
    数据增强:旋转,高斯噪声
    '''
    enhance_img_list = [img]
    enhance_seg_list = [seg]

    gauss_img, gauss_seg = gauss_noise(img, seg)
    enhance_img_list.append(gauss_img)
    enhance_seg_list.append(gauss_seg)

    color_img, color_seg = random_color(img, seg)
    enhance_img_list.append(color_img)
    enhance_seg_list.append(color_seg)
    
    angles = np.random.uniform(-40, 40, 2)
    # angles = [90, 180, 270]

    for angle in angles:
        rotate_img, rotate_seg = rotate(img, seg, angle)
        rotate_gauss_img, rotate_gauss_seg = rotate(gauss_img, gauss_seg, angle)
        rotate_color_img, rotate_color_seg = rotate(color_img, color_seg, angle)

        enhance_img_list.append(rotate_img)
        enhance_img_list.append(rotate_gauss_img)
        enhance_img_list.append(rotate_color_img)

        enhance_seg_list.append(rotate_seg)
        enhance_seg_list.append(rotate_gauss_seg)
        enhance_seg_list.append(rotate_color_seg)

    return enhance_img_list, enhance_seg_list

def load_data(data_path):
    img_path = os.path.join(data_path, 'img')
    seg_path = os.path.join(data_path, 'seg')
    
    img_list, seg_list = [], []

    for i, img_name in enumerate(os.listdir(img_path)):

        seg_name = img_name[:-3] + 'png'
        
        img = Image.open(os.path.join(img_path, img_name))
        img = img.resize((512, 512), Image.NEAREST)

        seg = Image.open(os.path.join(seg_path, seg_name))
        seg = seg.resize((512, 512), Image.NEAREST)

        print(f'{img_name}, {seg_name} loaded.')

        enhance_img_list, enhance_seg_list = data_enhance(img, seg)
        img_list += enhance_img_list
        seg_list += enhance_seg_list

    return img_list, seg_list


if __name__ == '__main__':
    data_path = '../datasets/meter_deep_voc/'

    img_list, seg_list = load_data(data_path)
    
    img_save_path = os.path.join(data_path, 'JPEGImages')
    seg_save_path = os.path.join(data_path, 'SegmentationClass')
    segvis_save_path = os.path.join(data_path, 'SegmentationClassVisualization')

    # if os.path.exists(img_save_path):
    #     os.rename(img_save_path, os.path.join(data_path, 'img'))

    # if os.path.exists(seg_save_path):
    #     os.rename(seg_save_path, os.path.join(data_path, 'seg'))

    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    
    if not os.path.exists(seg_save_path):
        os.makedirs(seg_save_path)

    if not os.path.exists(segvis_save_path):
        os.makedirs(segvis_save_path)

    for i, img, seg in zip(range(1, len(img_list) + 1), img_list, seg_list):
        
        img_name = f'{i}.jpg'
        seg_name = f'{i}.png'
        segvis_name = f'{i}.png'

        img.save(os.path.join(img_save_path, img_name))
        seg.save(os.path.join(seg_save_path, seg_name))

        img_cv = cv2.imread(os.path.join(img_save_path, img_name))
        seg_cv = cv2.imread(os.path.join(seg_save_path, seg_name))
        segvis = cv2.add(seg_cv, img_cv)
        cv2.imwrite(os.path.join(segvis_save_path, segvis_name), segvis)
        
        print(f'({i}/{len(img_list)}) {img_name}, {seg_name} written.')

    
        