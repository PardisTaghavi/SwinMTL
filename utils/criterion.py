# ------------------------------------------------------------------------------
# The original code is from GLPDepth (https://github.com/vinvino02/GLPDepth)
# moddified by Pardis Taghavi(taghavi.pardis@gmail.com)
# For non-commercial purpose only (research, evaluation etc).
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        loss = nn.CrossEntropyLoss()(pred, target)
        return loss
    
class SmoothLoss(nn.Module): 
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, depth, img):
        if len(depth.shape)==3:
            depth_=depth.unsqueeze(1)
        elif len(depth.shape)==4:
            depth_=depth
        elif len(depth.shape)==2:
            depth_=depth.unsqueeze(0).unsqueeze(0)
        
        grad_depth_x = torch.abs(depth_[:, :, :, :-1] - depth_[:, :, :, 1:])
        grad_depth_y = torch.abs(depth_[:, :, :-1, :] - depth_[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
        grad_img_y= torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)

        grad_depth_x *= torch.exp(-grad_img_x)
        grad_depth_y *= torch.exp(-grad_img_y)

        return grad_depth_x.mean() + grad_depth_y.mean()


