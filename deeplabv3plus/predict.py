import os
import os.path as osp
import numpy as np

import cv2

from paddlex import transforms as T
import paddlex as pdx

from numba import jit

@jit(nopython=True)
def label2png(seg):
    seg_png = np.zeros((512, 512, 3))

    for i in range(512):
        for j in range(512):
            if seg[i, j] == 1:
                seg_png[i, j, 2] = 128
            elif seg[i, j] == 2:
                seg_png[i, j, 1] = 128
    return seg_png

