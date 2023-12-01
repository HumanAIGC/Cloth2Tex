# -*- coding: utf-8 -*-

"""
    @date:  2023.01.29  week4  Sunday
    @func:  tipping point
"""

import numpy as np
import cv2


def extract_tipping_point(imgray):
    
    # 0) 读取图像
    contours, hierarchy = cv2.findContours(image=imgray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    
    # 1) 取y最小的值.

    candidate_set = []

    for item in contours:

        if len(item.squeeze().shape) == 1:
            if len(candidate_set) < 2:
                candidate_set.append([item.squeeze()[0], item.squeeze()[1]])
            elif len(candidate_set) == 2:
                x, y = item.squeeze()[0], item.squeeze()[1]
                if y < candidate_set[0][1]:
                    pop_idx = 0
                    candidate_set.append([x, y])
                elif y < candidate_set[1][1]:
                    pop_idx = 1
                    candidate_set.append([x, y])
                else:
                    pop_idx = None

                if pop_idx is not None:
                    candidate_set.pop(pop_idx)
                    pop_idx = None
        elif len(item.squeeze().shape) > 1:
            if len(candidate_set) == 2:
                item = item.squeeze()
                for each_item in item:
                    x, y = each_item[0], each_item[1]
                    if y < candidate_set[0][1]:
                        pop_idx = 0
                        candidate_set.append([x, y])
                    elif y < candidate_set[1][1]:
                        pop_idx = 1
                        candidate_set.append([x, y])
                    else:
                        pop_idx = None

                    if pop_idx is not None:
                        candidate_set.pop(pop_idx)
                        pop_idx = None
            elif len(candidate_set) < 2:
                item = item.squeeze()
                for each_item in item:
                    candidate_set.append([each_item.squeeze()[0], each_item.squeeze()[1]])

    return candidate_set