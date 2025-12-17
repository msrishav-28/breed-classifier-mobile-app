#!/usr/bin/env python3
"""
CBAM (Convolutional Block Attention Module) Integration for YOLOv8
Custom attention modules for enhanced feature learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelAttention(nn.Module):
    """Channel Attention Module of CBAM"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average pooling branch
        avg_out = self.fc(self.avg_pool(x))
        
        # Max pooling branch
        max_out = self.fc(self.max_pool(x))
        
        # Combine and apply sigmoid
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module of CBAM"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid
        out = self.conv(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    """Complete CBAM (Convolutional Block Attention Module)"""
    
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_attention(x)
        
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        
        return x

class CBAMBottleneck(nn.Module):
    """Bottleneck block with CBAM attention"""
    
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, k[0] // 2, groups=g, bias=False)
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, k[1] // 2, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        self.add = shortcut and c1 == c2
        
        # CBAM attention
        self.cbam = CBAM(c2)
    
    def forward(self, x):
        identity = x
        
        # First conv
        out = self.act(self.bn1(self.cv1(x)))
        
        # Second conv
        out = self.bn2(self.cv2(out))
        
        # Apply CBAM attention
        out = self.cbam(out)
        
        # Add residual connection
        if self.add:
            out += identity
        
        out = self.act(out)
        return out

class CBAMC2f(nn.Module):
    """C2f module with CBAM attention for YOLOv8"""
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        
        # Bottleneck layers with CBAM
        self.m = nn.ModuleList(
            CBAMBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )
        
        # Final CBAM attention
        self.cbam = CBAM(c2)
    
    def forward(self, x):
        # Initial convolution
        y = list(self.act(self.bn1(self.cv1(x))).split((self.c, self.c), 1))
        
        # Apply bottleneck layers
        for m in self.m:
            y.append(m(y[-1]))
        
        # Concatenate and final convolution
        out = self.act(self.bn2(self.cv2(torch.cat(y, 1))))
        
        # Apply final CBAM attention
        out = self.cbam(out)
        
        return out

class CBAMConv(nn.Module):
    """Standard convolution with CBAM attention"""
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        
        # CBAM attention
        self.cbam = CBAM(c2)
    
    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.cbam(x)
        return x

class CBAMSPPFast(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) with CBAM attention"""
    
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        
        # CBAM attention
        self.cbam = CBAM(c2)
    
    def forward(self, x):
        x = self.act(self.bn1(self.cv1(x)))
        
        # Apply max pooling multiple times
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        
        # Concatenate and final convolution
        out = self.act(self.bn2(self.cv2(torch.cat([x, y1, y2, y3], 1))))
        
        # Apply CBAM attention
        out = self.cbam(out)
        
        return out

class ECA(nn.Module):
    """Efficient Channel Attention (ECA) - Lightweight alternative to CBAM"""
    
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        
        # Calculate kernel size adaptively
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Global average pooling
        y = self.avg_pool(x)
        
        # 1D convolution
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        
        # Apply sigmoid and multiply
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        
        # Scale
        return x * y.expand_as(x)

class AttentionGate(nn.Module):
    """Attention Gate for feature fusion"""
    
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

def replace_yolo_modules_with_cbam(model):
    """Replace standard YOLOv8 modules with CBAM-enhanced versions"""
    
    for name, module in model.named_modules():
        # Replace C2f modules with CBAM versions
        if isinstance(module, torch.nn.Module) and 'C2f' in str(type(module)):
            # Get module parameters
            c1 = module.cv1.in_channels
            c2 = module.cv2.out_channels
            n = len(module.m)
            
            # Replace with CBAM version
            cbam_module = CBAMC2f(c1, c2, n)
            
            # Copy weights if possible
            try:
                cbam_module.load_state_dict(module.state_dict(), strict=False)
            except:
                pass
            
            # Replace module
            parent = model
            for attr in name.split('.')[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, name.split('.')[-1], cbam_module)
    
    return model

# Export functions for easy import
__all__ = [
    'CBAM',
    'ChannelAttention', 
    'SpatialAttention',
    'CBAMBottleneck',
    'CBAMC2f',
    'CBAMConv',
    'CBAMSPPFast',
    'ECA',
    'SEBlock',
    'AttentionGate',
    'replace_yolo_modules_with_cbam'
]