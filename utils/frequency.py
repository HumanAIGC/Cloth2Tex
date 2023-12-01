# -*- coding: utf-8 -*-
"""
    @date:  2023.02.07  week6  Tuesday
    @func:  freq & shift
    @ref1:  https://stackoverflow.com/questions/65680001/fft-loss-in-pytorch
    @ref2:  https://zhuanlan.zhihu.com/p/422962341
"""

import torch
import numpy as np
import cv2

def extract_ampl_phase(input_img):
    
    fft_img = torch.fft.rfftn(input_img.clone())
    fft_im = torch.stack((fft_img.real, fft_img.imag), -1)
    
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp) # amplitude
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0]) # phase
    return fft_amp, fft_pha


if __name__ == "__main__":
    a1 = torch.randn(1,3,512,512)
    aaa, bbb = extract_ampl_phase(a1)