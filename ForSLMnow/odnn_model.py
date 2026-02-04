import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import savemat
import numpy as np, os
from scipy.io import savemat
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
import time
from datetime import datetime
import json
import os
import pandas as pd
import torch.nn.functional as F
import mat73
import h5py # save the data as MATLAB 7.3
import random

#% padding funktion
def propagation(E, z, lam,layer_size,pixel_size,device):
        # Convert input to PyTorch tensor and move to GPU
        # E = torch.tensor(E, dtype=torch.complex64, device=device)
        E = E.clone().detach().to(dtype=torch.complex64, device=device)
        
        fft_c = torch.fft.fft2(E)
        c = torch.fft.fftshift(fft_c)
        
        fx = torch.fft.fftshift(torch.fft.fftfreq(layer_size, d=pixel_size)).to(device)
        fxx, fyy = torch.meshgrid(fx, fx, indexing='ij')

        argument = (2 * np.pi) ** 2 * ((1. / lam) ** 2 - fxx ** 2 - fyy ** 2)
        
        tmp = torch.sqrt(torch.abs(argument))
        kz = torch.where(argument >= 0, tmp, 1j * tmp)
        
        E = torch.fft.ifft2(torch.fft.ifftshift(c * torch.exp(1j * kz * z)))
        return E

def complex_pad(E, pad_h, pad_w):
    # E: (..., H, W) complex64
    Er = torch.view_as_real(E)                         # (..., H, W, 2)
    Er_pad = F.pad(Er, (0, 0, pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
    return torch.view_as_complex(Er_pad.contiguous())  # 确保存储连续

def complex_crop(E_pad, H, W, pad_h, pad_w):
    return E_pad[..., pad_h:pad_h+H, pad_w:pad_w+W].contiguous()

def make_pad_slices(H, W, padding_ratio=None, pad_px=None):
    """根据比例或像素给出 pad_h/pad_w 以及中心切片"""
    if pad_px is None:
        pad_h = int(round(H * padding_ratio))
        pad_w = int(round(W * padding_ratio))
    else:
        pad_h = pad_w = int(pad_px)
    sl_h = slice(pad_h, pad_h + H)
    sl_w = slice(pad_w, pad_w + W)
    return pad_h, pad_w, (sl_h, sl_w)

def complex_pad_asymm(E, pad_top, pad_bottom, pad_left, pad_right):
    # E: (..., H, W) complex64
    Er = torch.view_as_real(E)  # (..., H, W, 2)
    Er_pad = F.pad(Er, (0, 0, pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return torch.view_as_complex(Er_pad.contiguous())

class Propagation(nn.Module):
    """
    自由传播层
    """
    def __init__(self, units, dx, lam, z, device, pad_px=0):
        super().__init__()
        self.units  = units     # 原始 H=W=units
        self.dx     = dx
        self.lam    = lam
        self.z      = z
        self.pad_px = int(pad_px)

        self.register_buffer("kz_base", self._make_kz(units, dx, lam, device))

        if self.pad_px > 0:
            units_pad = units + 2 * self.pad_px
            self.register_buffer("kz_pad", self._make_kz(units_pad, dx, lam, device))
        else:
            self.kz_pad = None

    def _make_kz(self, N, dx, lam, device):
        fx = torch.fft.fftshift(torch.fft.fftfreq(N, d=dx)).to(device)
        fxx, fyy = torch.meshgrid(fx, fx, indexing='ij')
        argument = (2 * torch.pi) ** 2 * ((1. / lam) ** 2 - fxx ** 2 - fyy ** 2)
        tmp = torch.sqrt(torch.abs(argument))
        kz = torch.where(argument >= 0, tmp, 1j * tmp).to(torch.complex64)
        return kz

    def _propagate(self, E, kz, z):
        E = E.to(torch.complex64)
        C = torch.fft.fftshift(torch.fft.fft2(E))
        return torch.fft.ifft2(torch.fft.ifftshift(C * torch.exp(1j * kz * z)))

    def forward(self, inputs):
       
        assert inputs.is_complex(), "Propagation expects complex64 inputs."
        B, C, H, W = inputs.shape

        if self.pad_px > 0:
            p = self.pad_px
            # 去掉通道做 padding
            Ein = complex_pad(inputs.squeeze(1), p, p)           # (B, H+2p, W+2p)
            Eout = self._propagate(Ein, self.kz_pad, self.z)     # 传播在大画布
            Eout = complex_crop(Eout, H, W, p, p).unsqueeze(1)   # 裁回并补通道 (B,1,H,W)
            return Eout
        else:
            
            Eout = self._propagate(inputs, self.kz_base, self.z)
            return Eout

class DiffractionLayer(nn.Module):
    def __init__(self, units, dx, lam, z, device, pad_px=0):
        super().__init__()
        self.units = units      # 原始 H=W=units
        self.dx    = dx
        self.lam   = lam
        self.z     = z
        self.pad_px = pad_px    # 每边像素 padding

        self.phase = nn.Parameter(torch.randn(units, units, dtype=torch.float32))

        self.register_buffer("kz_base", self._make_kz(units, dx, lam, device))
        if pad_px > 0:
            units_pad = units + 2*pad_px
            self.register_buffer("kz_pad", self._make_kz(units_pad, dx, lam, device))
        else:
            self.kz_pad = None

    def _make_kz(self, N, dx, lam, device):
        fx = torch.fft.fftshift(torch.fft.fftfreq(N, d=dx)).to(device)
        fxx, fyy = torch.meshgrid(fx, fx, indexing='ij')
        argument = (2*torch.pi)**2 * ((1./lam)**2 - fxx**2 - fyy**2)
        tmp = torch.sqrt(torch.abs(argument))
        kz = torch.where(argument >= 0, tmp, 1j*tmp).to(torch.complex64)
        return kz

    def _propagate(self, E, kz, z):
        E = E.to(torch.complex64)
        C = torch.fft.fftshift(torch.fft.fft2(E))
        return torch.fft.ifft2(torch.fft.ifftshift(C * torch.exp(1j * kz * z)))

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        phase_c = torch.exp(1j * self.phase.to(inputs.device, dtype=torch.float32)).to(torch.complex64)

        if self.pad_px > 0:
            pad_h = pad_w = self.pad_px
            Ein = complex_pad(inputs.squeeze(1), pad_h, pad_w)     # 光场外圈为0 
             
            phase_big = torch.ones(H+2*pad_h, W+2*pad_w, dtype=torch.complex64, device=inputs.device) #相位圈为1
            #phase_big = torch.zeros(H+2*pad_h, W+2*pad_w, dtype=torch.complex64, device=inputs.device) #相位圈为0，好像无所谓
            phase_big[pad_h:pad_h+H, pad_w:pad_w+W] = phase_c
            Ein = Ein * phase_big 

            Eout = self._propagate(Ein, self.kz_pad, self.z)

            # 5) 裁回原始尺寸
            Eout = complex_crop(Eout, H, W, pad_h, pad_w).unsqueeze(1)  # (B,1,H,W)
            return Eout
        else:
            Ein = inputs * phase_c.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            Eout = self._propagate(Ein, self.kz_base, self.z)
            return Eout


class RegressionDetector(nn.Module):
    def __init__(self):
        super(RegressionDetector, self).__init__()

    def forward(self, inputs):
        # Compute intensity of the field
        return torch.square(torch.abs(inputs)) #取了平方


class D2NNModel(nn.Module):
        def __init__(self, num_layers, layer_size, z_layers, z_prop, pixel_size, wavelength, device,
                    padding_ratio=0.5, z_input_to_first=0.0):
            super().__init__()
            pad_px = int(round(layer_size * padding_ratio))
            #加上了第一层的传播
            self.pre_propagation = Propagation(layer_size, pixel_size, wavelength, z_input_to_first, device, pad_px=pad_px)
            self.layers = nn.ModuleList([
                DiffractionLayer(layer_size, pixel_size, wavelength, z_layers, device, pad_px=pad_px)
                for _ in range(num_layers)
            ])
            self.propagation = Propagation(layer_size, pixel_size, wavelength, z_prop, device, pad_px=pad_px)
            self.regression  = RegressionDetector()  

        def forward(self, x):
            x = self.pre_propagation(x)
            for layer in self.layers:
                x = layer(x)            #  pad->prop->crop
            x = self.propagation(x)     #  pad->prop->crop
            x = self.regression(x)     
            return x
        