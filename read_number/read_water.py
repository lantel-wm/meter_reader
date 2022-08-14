import os
import cv2
import numpy as np

from numba import jit

LABEL_DICT = {
    'background': 0, 
    'water': 1,
    'meter': 2,
}

@jit(nopython=True)
def label2png(seg):
    seg_png = np.zeros((seg.shape[0], seg.shape[1], 3))

    for i in range(seg_png.shape[0]):
        for j in range(seg_png.shape[1]):
            if seg[i, j] == 1:
                seg_png[i, j, 2] = 128
            elif seg[i, j] == 2:
                seg_png[i, j, 1] = 128
    return seg_png

@jit(nopython=True)
def get_bin_meter(seg_label):
    binaried_data = np.zeros(seg_label.shape[0])

    for row in range(seg_label.shape[0]):
        bg_cnt = 0
        for col in range(seg_label.shape[1]):
            if bg_cnt == 3:
                binaried_data[row] = 1
                break
            if seg_label[row, col] != 2:
                bg_cnt += 1

    return binaried_data


@jit(nopython=True)
def get_bin_water(seg_label):
    binaried_data = np.zeros(seg_label.shape[0])

    for row in range(seg_label.shape[0]):
        water_cnt = 0
        for col in range(seg_label.shape[1]):
            if water_cnt == 5:
                binaried_data[row] = 1
                break
            if seg_label[row, col] == 1: # LABEL_DICT['water']:
                water_cnt += 1

    return binaried_data
                
def get_water_location(binaried_water):
    loc = 0
    for i in range(len(binaried_water) - 1):
        if binaried_water[i] == 0 and binaried_water[i + 1] == 1:
            loc = i
            break
    return loc

def get_meter_location(binaried_meter):
    meter_upper, meter_lower = 0, 0
    for i in range(len(binaried_meter) - 1):
        if binaried_meter[i] == 0 and binaried_meter[i + 1] == 1:
            meter_upper = i
        elif binaried_meter[i] == 1 and binaried_meter[i + 1] == 0:
            meter_lower = i + 1
        
        if meter_lower != 0 and meter_upper != 0:
            break
    return meter_upper, meter_lower

@jit(nopython=True)
def get_water_bar(seg_label, water_location, meter_upper, meter_lower):

    water_bar = np.zeros((seg_label.shape[0], seg_label.shape[1], 3))
    water_bar_height = meter_lower - meter_upper
    water_bar_width = 0.05 * water_bar_height


    for row in range(meter_upper, meter_lower):
        for col in range(seg_label.shape[1] - water_bar_width, seg_label.shape[1]):
            # air
            if row <= water_location:
                water_bar[row, col, 2] = 255 # int('FF', 16)
                water_bar[row, col, 1] = 255 # int('FF', 16)
                water_bar[row, col, 0] = 240 # int('F0', 16)
            # water
            else:
                water_bar[row, col, 2] = 102 # int('66', 16)
                water_bar[row, col, 1] = 153 # int('99', 16)
                water_bar[row, col, 0] = 204 # int('CC', 16)
    return water_bar

def get_water_percent(water_location, meter_upper, meter_lower):
    water_percent = (meter_lower - water_location) / (meter_lower - meter_upper + 1e-5) * 100
    return round(water_percent)

def read_water(img, segmenter):
    '''
    water: 水位
    meter: 水表
    '''
    seg_result = segmenter.predict(img)
    seg_label = seg_result['label_map']
    seg_png = label2png(seg_label)

    binaried_meter = get_bin_meter(seg_label)
    binaried_water = get_bin_water(seg_label)

    water_location = get_water_location(binaried_water)
    meter_upper, meter_lower = get_meter_location(binaried_meter)
    water_bar = get_water_bar(seg_label, water_location, meter_upper, meter_lower)
    water_percent = get_water_percent(water_location, meter_upper, meter_lower)
    
    return seg_png, water_bar, water_percent, meter_upper, meter_lower