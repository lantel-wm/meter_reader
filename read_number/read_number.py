import os
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

from numba import jit

from yolov5.utils.general import LOGGER
from deeplabv3plus.predict import label2png

# 读数后处理中有把圆形表盘转成矩形的操作，矩形的宽即为圆形的外周长
# 因此要求表盘图像大小为固定大小，这里设置为[512, 512]
METER_SHAPE = (512, 512)  # 高x宽
# 圆形表盘的中心点
CIRCLE_CENTER = (256, 256)  # 高x宽
# 圆形表盘的半径
CIRCLE_RADIUS = 250
# 圆周率
PI = 3.1415926536
# 在把圆形表盘转成矩形后矩形的高
# 当前设置值约为半径的一半，原因是：圆形表盘的中心区域除了指针根部就是背景了
# 我们只需要把外围的刻度、指针的尖部保存下来就可以定位出指针指向的刻度
RECTANGLE_HEIGHT = 110
# 矩形表盘的宽，即圆形表盘的外周长
RECTANGLE_WIDTH = 1570
# 当前案例中只使用了两种类型的表盘，第一种表盘的刻度根数为50
# 第二种表盘的刻度根数为32。因此，我们通过预测的刻度根数来判断表盘类型
# 刻度根数超过阈值的即为第一种，否则是第二种
# TYPE_THRESHOLD = 40
# 两种表盘的配置信息，包含每根刻度的值，量程，单位
METER_CONFIG = {
    'scale_interval_value': 0.1,
    'range': 4.0,
    'scale_range': 40,
    'unit': "(kPa)",
}

# def get_label(img_np):
#     '''
#     输入图片(numpy array),返回label类型
#     0: background
#     1: pointer
#     2: scale
#     '''
#     label_np = np.zeros(METER_SHAPE)

#     for i in range(METER_SHAPE[0]):
#         for j in range(METER_SHAPE[1]):
#             if img_np[i, j, 0] == 0 and img_np[i, j, 1] == 0 and img_np[i, j, 2] == 0:
#                 label_np[i, j] =0
#             elif img_np[i, j, 0] == 128 and img_np[i, j, 1] == 0 and img_np[i, j, 2] == 0:
#                 label_np[i, j] =0
#             elif img_np[i, j, 0] == 0 and img_np[i, j, 1] == 128 and img_np[i, j, 2] == 0:
#                 label_np[i, j] =0

# def white_background(seg):
#     for i in range(seg.shape[0]):
#         for j in range(seg.shape[1]):
#             if seg[i, j, 0] == 0 and seg[i, j, 1] == 0 and seg[i, j, 2] == 0:
#                 seg[i, j, 0] = 255
#                 seg[i, j, 1] = 255
#                 seg[i, j, 2] = 255
#             elif seg[i, j, 0] == 0 and seg[i, j, 1] == 128 and seg[i, j, 2] == 128:
#                 seg[i, j, 0] = 255
#                 seg[i, j, 1] = 255
#                 seg[i, j, 2] = 255
#     return seg

@jit(nopython=True)
def circle_to_rectangle(seg):
    '''
    将圆形表盘的预测结果label_map转换成矩形

    圆形到矩形的计算方法：
        因本案例中两种表盘的刻度起始值都在左下方，故以圆形的中心点为坐标原点，
        从-y轴开始逆时针计算极坐标到x-y坐标的对应关系：
            x = r + r * cos(theta)
            y = r - r * sin(theta)
        注意：
            1. 因为是从-y轴开始逆时针计算，所以r * sin(theta)前有负号。
            2. 还是因为从-y轴开始逆时针计算，所以矩形从上往下对应圆形从外到内，
                可以想象把圆形从-y轴切开再往左右拉平时，圆形的外围是上面，內围在下面。

    参数：
        seg (cv2 image)：分割模型的预测结果。

    返回值：
        rectangle_meters (np.array)：矩形表盘的预测结果label_map。

    '''
    rectangle_meter = np.zeros((RECTANGLE_HEIGHT, RECTANGLE_WIDTH, 3), dtype=np.uint8)
    for row in range(RECTANGLE_HEIGHT):
        for col in range(RECTANGLE_WIDTH):
            theta = PI * 2 * (col + 1) / RECTANGLE_WIDTH
            # 矩形从上往下对应圆形从外到内
            rho = CIRCLE_RADIUS - row - 1
            y = int(CIRCLE_CENTER[0] + rho * math.cos(theta) + 0.5)
            x = int(CIRCLE_CENTER[1] - rho * math.sin(theta) + 0.5)
            rectangle_meter[row, col, :] = seg[y, x, :]
    return rectangle_meter

@jit(nopython=True)
def rectangle_to_line(rectangle_meter):
    '''
    从矩形表盘的预测结果中提取指针和刻度预测结果并沿高度方向压缩成线状格式。

    参数：
        rectangle_meters (np.array)：矩形表盘的预测结果label_map。

    返回：
        line_scales (np.array)：刻度的线状预测结果。
        line_pointers (np.array)：指针的线状预测结果。

    '''
    height, width = rectangle_meter.shape[0:2]
    line_scale = np.zeros((width), dtype=np.uint8)
    line_pointer = np.zeros((width), dtype=np.uint8)
    for col in range(width):
        for row in range(height):
            if rectangle_meter[row, col, 0] == 0 and rectangle_meter[row, col, 1] == 0 and rectangle_meter[row, col, 2] == 128:
                line_pointer[col] += 1
            elif rectangle_meter[row, col, 1] == 128:
                line_scale[col] += 1
    return line_scale, line_pointer

def mean_binarization(data):
    """对图像进行均值二值化操作

    参数：
        data (np.array)：待二值化的数组。

    返回：
        binaried_data (np.array)：二值化后的数组。

    """
    binaried_data = data
    mean_data = np.mean(data)
    width = data.shape[0]
    for col in range(width):
        if data[col] < mean_data:
            binaried_data[col] = 0
        else:
            binaried_data[col] = 1
    return binaried_data

def locate(data_type, line_data):
    """
    在线状预测结果中找到指针或每根刻度的中心位置

    参数：
        data_type (str) : pointer 或者 scale 
        line_data (np.array)：批量的二值化后的刻度线状预测结果。

    返回：
        scale_locations (list[(location, count)])：各图像中每根刻度的中心位置。

    """

    width = line_data.shape[0]
    find_start = False
    one_scale_start = 0
    one_scale_end = 0
    locations = list()
    for j in range(width - 1):
        if line_data[j] > 0 and line_data[j + 1] > 0:
            if find_start == False:
                one_scale_start = j
                find_start = True
        if find_start:
            if line_data[j] == 0 and line_data[j + 1] == 0:
                one_scale_end = j - 1
                one_scale_location = (one_scale_start + one_scale_end) / 2
                one_scale_cnt = one_scale_end - one_scale_start
                if data_type == 'pointer':
                    # 可能识别出多个pointer, 取像素点最多的那个
                    locations.append((one_scale_location, one_scale_cnt))
                elif data_type == 'scale':
                    locations.append(one_scale_location)
                one_scale_start = 0
                one_scale_end = 0
                find_start = False

    
    if data_type == 'scale':
        return locations
    elif data_type == 'pointer':
        if len(locations) == 0:
            return -1
        else:
            # 可能识别出多个pointer, 取像素点最多的那个
            max_cnt = 0
            pointer_location = 0
            for pointer_loc in locations:
                loc = pointer_loc[0]
                cnt = pointer_loc[1]
                if cnt > max_cnt:
                    max_cnt = cnt
                pointer_location = loc
            return pointer_location

def get_relative_location(scale_locations, pointer_location):
    """
    找到指针指向了第几根刻度

    参数：
        scale_locations (np.array)：每根刻度的中心点位置。
        pointer_locations (int)：指针的中心点位置。

    返回：
        pointed_scales (float)：指向的刻度位置
    """

    # pointed_scales = list()
    if pointer_location == -1:
        return -1, -1
    num_scales = len(scale_locations)
    pointed_scale = -1
    pointed_scale_idx = 0
    if num_scales > 0:
        for i in range(num_scales - 1):
            if scale_locations[i] <= pointer_location and pointer_location < scale_locations[i + 1]:
                pointed_scale = i + (pointer_location - scale_locations[i]) / (scale_locations[i + 1] - scale_locations[i] + 1e-05) + 1
                if pointer_location - scale_locations[i] <= scale_locations[i + 1] - pointer_location:
                    pointed_scale_idx = i
                else:
                    pointed_scale_idx = i + 1
    if pointer_location >= scale_locations[-1]:
        pointer_location = scale_locations[-1]
        pointed_scale = METER_CONFIG['scale_range']
        pointed_scale_idx = METER_CONFIG['scale_range']

    if pointer_location <= scale_locations[0]:
        pointer_location = scale_locations[0]
        pointed_scale = 0
        pointed_scale_idx = 0
    # result = {'num_scales': num_scales, 'pointed_scale': pointed_scale}
    # pointed_scales.append(result)
    return pointed_scale, pointed_scale_idx

def calculate_reading(pointed_scale):
    """
    根据刻度的间隔值和指针指向的刻度根数计算表盘的读数
    """
    reading = pointed_scale * METER_CONFIG['scale_interval_value']
    return reading

def calc_diff(scale_locations):
    '''
    计算二阶差分数组
    二阶差分能够反映每个点与相邻两个点的中点的偏离程度
    '''
    scale_locations = np.array(scale_locations)
    max_scale, min_scale = max(scale_locations), min(scale_locations)

    # 将scale_locations的取值范围映射到 [1, 40]

    scale_locations = (scale_locations - min_scale) / (max_scale - min_scale) * (METER_CONFIG['scale_range'] - 1) + 1
    scale_location_diff = np.zeros(len(scale_locations))
    scale_location_diff2 = np.zeros(len(scale_locations))

    for i in range(len(scale_location_diff) - 1):
                        scale_location_diff[i + 1] = scale_locations[i + 1] - scale_locations[i]

    for i in range(1, len(scale_location_diff2) - 1):
        scale_location_diff2[i] = scale_location_diff[i + 1] - scale_location_diff[i] 

    scale_location_diff2 -= np.mean(scale_location_diff2)  
    # 第二根刻度特殊处理
    scale_location_diff2[1] = 0 

    return scale_location_diff2

def location_correction(scale_locations, pointer_location):
    '''
    根据location数组的二阶差分对异常值进行纠正
    case 1:
        少识别了刻度, 在二阶差分 >=1(0.9) 的两个刻度位置中间插入一个刻度
    case 2:
        多识别了刻度, 把二阶差分 >= 0.5(0.4) 的两个相隔刻度中间的刻度删除
    '''
    num_scale = len(scale_locations)
    scale_location_diff2 = calc_diff(scale_locations)

    # 少识别了刻度
    if num_scale == METER_CONFIG['scale_range'] - 1: # 39
        for i in range(1, num_scale - 1):
            if abs(scale_location_diff2[i]) >= 0.9 and abs(scale_location_diff2[i + 1]) >= 0.9:# and (scale_location_diff2[i] == max_diff2 or scale_location_diff2[i + 1] == max_diff2):
                add_scale = 0.5 * (scale_locations[i] + scale_locations[i + 1])
                scale_locations.insert(i + 1, add_scale)
                break
    
    # 多识别了刻度
    if num_scale == METER_CONFIG['scale_range'] + 1: # 41
        for i in range(1, num_scale - 2):
            if abs(scale_location_diff2[i]) >= 0.4 and abs(scale_location_diff2[i + 2]) >= 0.4:
                del scale_locations[i + 1]
                break

    # 如果校正后的刻度数量仍然错误, 扔出警告
    if len(scale_locations) != METER_CONFIG['scale_range']: # 40
        LOGGER.warning(f'Correction failed! Return -1')
        return scale_locations
    # 校正后的刻度数量正确
    else:
        # 重新读取指针和刻度的相对位置
        pointed_scale, pointed_idx = get_relative_location(scale_locations, pointer_location)
        # 重新计算二阶差分
        scale_location_diff2 = calc_diff(scale_locations)

        # 校正与指针相距最近的刻度的位置
        if (pointed_idx > 1 
            and pointed_idx < METER_CONFIG['scale_range'] 
            and abs(scale_location_diff2[pointed_idx]) >= 0.15):

            # 将与指针相距最近的刻度校正为相邻两个刻度的中点
            scale_locations[pointed_idx] = \
                0.5 * (scale_locations[pointed_idx + 1] + scale_locations[pointed_idx - 1])

        # LOGGER.info('Correction success!')
        return scale_locations

def predict(seg, seg_label=None, erode_kernel=4):
    '''
    从语义分割结果seg图片中读数
    '''
    # 识别失败标志
    fail_flag = False

    # 圆形表盘变换为矩形表盘
    rectangle_meter = circle_to_rectangle(seg)
    # 矩形图像沿指针方向压缩成一维数组
    line_scale, line_pointer = rectangle_to_line(rectangle_meter)

    # 一维数组二值化
    binaried_scale = mean_binarization(line_scale)
    binaried_pointer = mean_binarization(line_pointer)

    # 每个连续的 1 的区间中点识别为一个指针
    scale_locations = locate('scale', binaried_scale)
    pointer_location = locate('pointer', binaried_pointer)

    # 对指针附近的刻度进行校正
    if len(scale_locations) >= 39 and len(scale_locations) <= 41 and pointer_location != -1:
        scale_locations = location_correction(scale_locations, pointer_location)
    else:
        LOGGER.info(f'Scale recognition failed! Return -1')
        fail_flag = True
        
    
    if scale_locations is not None and not fail_flag:
        # 校正后进行读数
        pointed_scale, pointed_scale_idx = get_relative_location(scale_locations, pointer_location)
        meter_reading = calculate_reading(pointed_scale)
    else:
        meter_reading = -1

    return fail_flag, meter_reading, rectangle_meter, binaried_scale, binaried_pointer, scale_locations, pointer_location



def read_number_from_file(seg_path, img_path):
    '''
    从指定目录的语义分割图片中读数
    '''
    res = []
    scale = []
    pointer = []

    for seg_name in os.listdir(seg_path):
        img_name = seg_name[:-3] + 'jpg'
        img_file = os.path.join(img_path, img_name)
        img = cv2.imread(img_file)

        seg_file = os.path.join(seg_path, seg_name)
        seg = cv2.imread(seg_file)

        number, scale_locations, pointer_location = predict(seg)

        cv2.imshow(f'meter_reading: {number}', img)
        cv2.waitKey(0)
        # print('meter_reading: ', number)

        res.append(f'{seg_file}, {number}')
        scale.append(scale_locations)
        pointer.append(pointer_location)

    return res, scale, pointer

def read_number(img, segmenter):
    '''
    yolov5 裁剪出bbox后的图片接口, 返回:
    fail_flag (bool): 默认为False,为True说明识别出现问题,读数为-1或误差较大
    meter_reading (float): 读数
    seg_copy (cv2 image): 语义分割结果的备份
    rectangle_meter (cv2 image): 圆形表盘变换为矩形的结果
    binaried_scale (np.array): 二值化的刻度
    binaried_pointer (np.array): 二值化的指针
    scale_locations (list[int]): 识别到的指针位置
    pointer_location (int): 识别到的指针位置

    '''
    img_shape = img.shape
    if img_shape != (512, 512, 3):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    # seg = segmentation_predict(img, model, device, opts)
    seg_result = segmenter.predict(img)
    seg_label = seg_result['label_map']
    seg = label2png(seg_label)
    seg_copy = seg.copy()

    # cv2.imshow('seg', seg)
    # cv2.waitKey(0)
    # assert(0)

    # 调用预测接口
    (
        fail_flag,
        meter_reading, 
        rectangle_meter, 
        binaried_scale, 
        binaried_pointer, 
        scale_locations, 
        pointer_location,
    ) = predict(seg, seg_label)
    meter_reading = round(meter_reading, 2)

    # seg_copy = cv2.resize(seg_copy, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)

    return fail_flag, meter_reading, seg_copy, rectangle_meter, binaried_scale, binaried_pointer, scale_locations, pointer_location

    

if __name__ == '__main__':

    seg_path = '../DeepLabV3Plus-Pytorch/output'
    # seg_path = '../DeepLabV3Plus-Pytorch/datasets/data/VOCdevkit/VOC_meter/seg'
    img_path = '../DeepLabV3Plus-Pytorch/test'
    # img_path = '../DeepLabV3Plus-Pytorch/datasets/data/VOCdevkit/VOC_meter/img'

    res, scale, pointer = read_number_from_file(seg_path, img_path)
  

