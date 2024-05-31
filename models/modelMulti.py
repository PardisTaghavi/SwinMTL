# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The deconvolution code is based on Simple Baseline.
# (https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py)
#  moddified by Pardis Taghavi (taghavi.pardis@gmail.com)
# ------------------------------------------------------------------------------

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from mmcv.cnn import (
    build_conv_layer,
    build_norm_layer,
    build_upsample_layer,
    constant_init,
    normal_init,
)

from models.swin_transformer_v2 import SwinTransformerV2


def pretrained_weights_model(pretrained):
    #rename dict keys
    for k,v in pretrained.items():
        if "encoder" in k:
            new_k=k.replace("encoder.","")
            pretrained[new_k]=pretrained.pop(k)
        if "decoder" in k:
            #remove decoder
            pretrained.pop(k)

    return pretrained


class GLPDepth(nn.Module, PyTorchModelHubMixin):
    def __init__(self, args=None):
        super().__init__()
        self.max_depth = args.max_depth
        
        if 'tiny' in args.backbone:
            embed_dim = 96
            num_heads = [3, 6, 12, 24]
        elif 'base' in args.backbone:
            embed_dim = 128
            num_heads = [4, 8, 16, 32]
        elif 'large' in args.backbone:
            embed_dim = 192
            num_heads = [6, 12, 24, 48]
        elif 'huge' in args.backbone:
            embed_dim = 352
            num_heads = [11, 22, 44, 88]
        else:
            raise ValueError(args.backbone+" is not implemented, please add it in the models/model.py.")

        self.encoder = SwinTransformerV2(
            embed_dim=embed_dim,
            depths=args.depths,
            num_heads=num_heads,
            window_size=args.window_size,
            pretrain_window_size=args.pretrain_window_size,
            drop_path_rate=args.drop_path_rate,
            use_checkpoint=args.use_checkpoint,
            use_shift=args.use_shift,
        )
        self.num_classes = args.num_classes
        self.encoder.init_weights(pretrained=args.pretrained)
        
        channels_in = embed_dim*8
        channels_out = embed_dim
            
        self.decoder = Decoder(channels_in, channels_out, args)
        self.decoder.init_weights()

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))
        
        self.last_layer_seg = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, self.num_classes, kernel_size=3, stride=1, padding=1)
            )
        

        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
        
        for m in self.last_layer_seg.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

    def forward(self, x):      

        conv_feats = self.encoder(x)
        out = self.decoder(conv_feats[0]) 
        out_depth = self.last_layer_depth(out)
        out_depth = torch.sigmoid(out_depth) 

        #from log space to meters
        d_min=1e-3
        d_max=80
        out_depth = d_min * torch.exp(torch.log(torch.tensor(d_max / d_min)) * out_depth)
        out_seg = self.last_layer_seg(out)

        return {'pred_d': out_depth, 'pred_seg': out_seg}


class GLPDepthAttention(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.max_depth = args.max_depth
        
        if 'tiny' in args.backbone:
            embed_dim = 96
            num_heads = [3, 6, 12, 24]
        elif 'base' in args.backbone:
            embed_dim = 128
            num_heads = [4, 8, 16, 32]
        elif 'large' in args.backbone:
            embed_dim = 192
            num_heads = [6, 12, 24, 48]
        elif 'huge' in args.backbone:
            embed_dim = 352
            num_heads = [11, 22, 44, 88]
        else:
            raise ValueError(args.backbone+" is not implemented, please add it in the models/model.py.")

        self.encoder = SwinTransformerV2(
            embed_dim=embed_dim,
            depths=args.depths,
            num_heads=num_heads,
            window_size=args.window_size,
            pretrain_window_size=args.pretrain_window_size,
            drop_path_rate=args.drop_path_rate,
            use_checkpoint=args.use_checkpoint,
            use_shift=args.use_shift,
        )
        self.num_classes = args.num_classes

        self.encoder.init_weights(pretrained=args.pretrained)
        
        channels_in = embed_dim*8
        channels_out = embed_dim
            
        self.decoder = Decoder(channels_in, channels_out, args)
        self.decoder.init_weights()

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))
        
        self.last_layer_seg = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, self.num_classes, kernel_size=3, stride=1, padding=1)
            )
        
        embed_dim = 128
        num_heads = 8
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=False),
            nn.Linear(embed_dim, embed_dim)
        )
        self.layerNorm = nn.LayerNorm(embed_dim)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        

        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
                
        for m in self.last_layer_seg.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

    def forward(self, x):   

        conv_feats = self.encoder(x)
        out = self.decoder(conv_feats) 
        B, C, H, W = out.shape
        out = out.view(B, C, -1).permute(0, 2, 1) 
        out_=out

        d = self.linear1(out)
        d = self.layerNorm(d)

        s = self.linear1(out_) 
        s = self.layerNorm(s)
    
        d_=d
        #cross attention to output depth using segmentation as query
        d, _ = self.multihead_attn(s, d, d)         
            
        d   = self.layerNorm(d)
        d   = self.linear1(d) 
        d   = self.layerNorm(d)

        s, _ = self.multihead_attn(d_, s, s)
        s   = self.layerNorm(s)
        s   = self.linear1(s) 
        s   = self.layerNorm(s)

        d = d.permute(0, 2, 1).view(B, C, H, W)
        s = s.permute(0, 2, 1).view(B, C, H, W)  

        d = self.up(d) 
        d = self.up(d) 
        s = self.up(s) 
        s = self.up(s) 

        out_depth = self.last_layer_depth(d)
        out_depth = torch.sigmoid(out_depth) 
        out_seg = self.last_layer_seg(s)       

        return {'pred_d': out_depth, 'pred_seg': out_seg}
    


class GLPDepthAttention2(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.max_depth = args.max_depth
        
        if 'tiny' in args.backbone:
            embed_dim = 96
            num_heads = [3, 6, 12, 24]
        elif 'base' in args.backbone:
            embed_dim = 128
            num_heads = [4, 8, 16, 32]
        elif 'large' in args.backbone:
            embed_dim = 192
            num_heads = [6, 12, 24, 48]
        elif 'huge' in args.backbone:
            embed_dim = 352
            num_heads = [11, 22, 44, 88]
        else:
            raise ValueError(args.backbone+" is not implemented, please add it in the models/model.py.")

        self.encoder = SwinTransformerV2(
            embed_dim=embed_dim,
            depths=args.depths,
            num_heads=num_heads,
            window_size=args.window_size,
            pretrain_window_size=args.pretrain_window_size,
            drop_path_rate=args.drop_path_rate,
            use_checkpoint=args.use_checkpoint,
            use_shift=args.use_shift,
        )
        self.num_classes = args.num_classes

        self.encoder.init_weights(pretrained=args.pretrained)
        
        channels_in = embed_dim*8
        channels_out = embed_dim
            
        self.decoder = Decoder(channels_in, channels_out, args)
        self.decoder.init_weights()

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))
        
        self.last_layer_seg = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, self.num_classes, kernel_size=3, stride=1, padding=1)
            )
        
        embed_dim = 128
        num_heads = 8
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=False),
            nn.Linear(embed_dim, embed_dim)
        )
        self.layerNorm = nn.LayerNorm(embed_dim)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        

        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
                
        for m in self.last_layer_seg.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

    def forward(self, x):   

        conv_feats = self.encoder(x)
        out = self.decoder(conv_feats)
        B, C, H, W = out.shape
        print("out.shape:", out.shape)
        out = out.view(B, C, -1).permute(0, 2, 1) 

        d = self.linear1(out)
        d = self.layerNorm(d)
        s = self.linear1(out) 
        s = self.layerNorm(s)
    
        d, _ = self.multihead_attn(d, d, d) 
        d   = self.layerNorm(d)
        d   = self.linear1(d) 
        d   = self.layerNorm(d)

        s, _ = self.multihead_attn(s, s, s)
        s   = self.layerNorm(s)
        s   = self.linear1(s) 
        s   = self.layerNorm(s)

        d = d.permute(0, 2, 1).view(B, C, H, W)
        s = s.permute(0, 2, 1).view(B, C, H, W)  

        d = self.up(d) 
        d = self.up(d) 
        s = self.up(s) 
        s = self.up(s) 

        out_depth = self.last_layer_depth(d)
        out_depth = torch.sigmoid(out_depth) 
        out_seg = self.last_layer_seg(s)

        return {'pred_d': out_depth, 'pred_seg': out_seg}
    
class GLPDepthwCanny(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.max_depth = args.max_depth
        
        if 'tiny' in args.backbone:
            embed_dim = 96
            num_heads = [3, 6, 12, 24]
        elif 'base' in args.backbone:
            embed_dim = 128
            num_heads = [4, 8, 16, 32]
        elif 'large' in args.backbone:
            embed_dim = 192
            num_heads = [6, 12, 24, 48]
        elif 'huge' in args.backbone:
            embed_dim = 352
            num_heads = [11, 22, 44, 88]
        else:
            raise ValueError(args.backbone+" is not implemented, please add it in the models/model.py.")

        self.encoder = SwinTransformerV2(
            embed_dim=embed_dim,
            depths=args.depths,
            num_heads=num_heads,
            window_size=args.window_size,
            pretrain_window_size=args.pretrain_window_size,
            drop_path_rate=args.drop_path_rate,
            use_checkpoint=args.use_checkpoint,
            use_shift=args.use_shift,
        )
        self.num_classes = args.num_classes

        self.encoder.init_weights(pretrained=args.pretrained)
        
        channels_in = embed_dim*8
        channels_out = embed_dim
            
        self.decoder = Decoder(channels_in, channels_out, args)
        self.decoder.init_weights()

        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))
        
        self.last_layer_seg = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, self.num_classes, kernel_size=3, stride=1, padding=1)
            )
        
        embed_dim = 128
        num_heads = 8
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=False),
            nn.Linear(embed_dim, embed_dim)
        )
        self.layerNorm = nn.LayerNorm(embed_dim)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        

        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
                
        for m in self.last_layer_seg.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

    def forward(self, x):  

        canny = np.zeros((x.shape[0],1,x.shape[2],x.shape[3]))
        x= x.cpu().numpy().astype(np.uint8)*255
        x=x.transpose(0,2,3,1)

        for i in range(x.shape[0]):

            canny[i,:,:,:] = cv2.Canny(x[i,:,:,:], 100, 200)/255.0
            
        x=x.transpose(0,3,1,2)
        x=torch.from_numpy(x).cuda().float()/255.0

        x=torch.cat((x,canny),1)
        conv_feats = self.encoder(x)
        out = self.decoder(conv_feats)

        out = self.up(out)
        out = self.up(out)

        out_depth = self.last_layer_depth(out)
        out_depth = torch.sigmoid(out_depth) * self.max_depth
        out_seg = self.last_layer_seg(out)

        return {'pred_d': out_depth, 'pred_seg': out_seg}


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.deconv = args.num_deconv
        self.in_channels = in_channels
        
        self.deconv_layers = self._make_deconv_layer(
            args.num_deconv,
            args.num_filters,
            args.deconv_kernels,
        )
        
        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=args.num_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(
            build_norm_layer(dict(type='BN'), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, conv_feats):

        out = self.deconv_layers(conv_feats) 
        out = self.conv_layers(out) 
        out = self.up(out) 
        out = self.up(out) 

        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        
        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

class discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.Dlayers=nn.Sequential(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
                                      nn.LeakyReLU(0.2),
                                      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU(0.2),
                                      nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.LeakyReLU(0.2),
                                      nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.LeakyReLU(0.2),
                                      nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
                                      nn.Sigmoid()
                                      )
        for m in self.Dlayers.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

    def forward(self, x):
        
        outD= self.Dlayers(x) 
        return outD
    
class Critc(nn.Module):
   
    def __init__(self):
        super().__init__()

        self.Clayers=nn.Sequential(nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
                                      nn.LeakyReLU(0.2),
                                      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU(0.2),
                                      nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.LeakyReLU(0.2),
                                      nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
                                      nn.BatchNorm2d(512),
                                      nn.LeakyReLU(0.2),
                                      nn.Conv2d(512, 2, kernel_size=4, stride=1, padding=1)
                                      )
        for m in self.Clayers.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

    def forward(self, x):
            outC= self.Clayers(x) 
            return outC

    
class PatchCritic(nn.Module):

    ''' 
    input: depth map 1* 1* H* W and segmentation map 1* 1* H* W
    output size: 
    '''

    def __init__(self):
        super().__init__()

        self.Clayers=nn.Sequential(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(256),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
                                        )
        for m in self.Clayers.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

    def forward(self, x):

        outC= self.Clayers(x) 
        return outC