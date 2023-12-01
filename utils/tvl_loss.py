# coding: UTF-8
"""
    @date:  2023.03.17  week11  Friday
    @func:  Total Variation Loss
"""
import torch
import torch.nn as nn

class TVLoss(nn.Module):
    def __init__(self, weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        
         # 2023.03.29 +2nearest
        # h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum() + torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:])[:, :, ::2, :],2).sum()
        # w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum() + torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1])[:, :, :, ::2],2).sum()
        
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
    
class TVMaskLoss(nn.Module):
    def __init__(self, weight=1):
        super(TVMaskLoss,self).__init__()
        self.TVMaskLoss_weight = weight
        self.non_idx = None

    def forward(self, mask, x):
        if self.non_idx is None:
            non_idx = mask.nonzero()
            self.non_idx = non_idx.split(1, dim=1)
            
        tmp_mask = torch.ones(1,3,512,512).cuda()
        tmp_mask[self.non_idx] = 0 # 排除非UV区域.
        
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        
        x = x * tmp_mask
        
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        # h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        # w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        
        # 2023.03.29 +2nearest
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum() + torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:])[:, :, ::2, :],2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum() + torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1])[:, :, :, ::2],2).sum()
        return self.TVMaskLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]