# coding: UTF-8
"""
    @date:  2023.03.28  week13  Tuesday
    @func:  mask_iou loss from kaolin
    @ref:   https://github.com/NVIDIAGameWorks/kaolin/blob/889283262bc4b97c0251d6f5f2e3331531b3022b/kaolin/metrics/render.py
"""


import torch

def mask_iou(lhs_mask, rhs_mask):
    r"""Compute the Intersection over Union of two segmentation masks.
    Args:
        lhs_mask (torch.FloatTensor):
            A segmentation mask, of shape
            :math:`(\text{batch_size}, \text{height}, \text{width})`.
        rhs_mask (torch.FloatTensor):
            A segmentation mask, of shape
            :math:`(\text{batch_size}, \text{height}, \text{width})`.
    Returns:
        (torch.FloatTensor): The IoU loss, as a torch scalar.
    """
    batch_size, height, width = lhs_mask.shape
    assert rhs_mask.shape == lhs_mask.shape
    sil_mul = lhs_mask * rhs_mask
    sil_add = lhs_mask + rhs_mask
    iou_up = torch.sum(sil_mul.reshape(batch_size, -1), dim=1)
    iou_down = torch.sum((sil_add - sil_mul).reshape(batch_size, -1), dim=1)
    iou_neg = iou_up / (iou_down + 1e-10)
    mask_loss = 1.0 - torch.mean(iou_neg)
    return mask_loss