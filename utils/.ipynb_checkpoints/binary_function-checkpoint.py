# coding: UTF-8
"""
    @date:  2023.02.13  week7  Monday
    @func:  Binarize the rendered mask!!!
"""
import torch
import torch.nn as nn
import torch.nn.functional as f

from torch import Tensor
from torch.autograd import Function

class Binarize(Function):
    clip_value = 1

    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)

        output = inp.sign()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp: Tensor = ctx.saved_tensors[0]

        clipped = inp.abs() <= Binarize.clip_value

        output = torch.zeros(inp.size()).to(grad_output.device)
        output[clipped] = 1
        output[~clipped] = 0

        return output * grad_output
    
if __name__ == "__main__":
    binarize = Binarize.apply
    import pdb; pdb.set_trace()