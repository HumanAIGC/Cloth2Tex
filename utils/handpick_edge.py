# coding: UTF-8
"""
    @date:  2023.03.24  week12  Friday
    @ref:   https://github.com/captanlevi/Contour-Detection-Pytorch/blob/master/extract_contours.py
    @ref:   https://discuss.pytorch.org/t/setting-custom-kernel-for-cnn-in-pytorch/27176/3
"""

import torch
import numpy as np
import torch.nn.functional as F


def edge_extraction(mask):
    
    tmp_mask = mask.clone()
    # 1. [bs, 512, 512] --> [bs, 512, 512]边缘
    bs, H, W = mask.shape
    
    edge_mask = Filter_torch(tmp_mask)
    
    return edge_mask


def Filter_torch(arr):

    nb_channels = 1
    x = arr.unsqueeze(1)
    
    weights = torch.tensor([[1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.]]).cuda()
    weights = weights.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
    output = F.conv2d(x, weights,padding='same')

    ccc = 1 - output.lt(7).float() + 1 - output.gt(4).float()
    return 1 - ccc


if __name__ == "__main__":
    pass