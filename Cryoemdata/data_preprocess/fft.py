import numpy as np
import torch
from torch.fft import fftshift, fft2


def fft2_center(img, tensor=False):
    if tensor:
        return fftshift(fft2(fftshift(img, dim=(-1, -2))), dim=(-1, -2))
    else:
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, axes=(-1, -2))), axes=(-1, -2))


def fftn_center(img):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img)))


def ifftn_center(vol):
    vol = np.fft.ifftshift(vol)
    vol = np.fft.ifftn(vol)
    vol = np.fft.fftshift(vol)
    return vol


def ht2_center(img, tensor=False):
    f = fft2_center(img, tensor=tensor)
    return f.real - f.imag


def htn_center(img):
    f = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img)))
    return f.real - f.imag


def iht2_center(img):
    img = fft2_center(img)
    img /= (img.shape[-1] * img.shape[-2])
    return img.real - img.imag


def ihtn_center(vol):
    vol = np.fft.ifftshift(vol)
    vol = np.fft.fftn(vol)
    vol = np.fft.fftshift(vol)
    vol /= np.prod(vol.shape)
    return vol.real - vol.imag


def symmetrize_ht(ht, pre_allocated=False):
    if pre_allocated:
        resolution = ht.shape[-1] - 1
        sym_ht = ht
    else:
        if len(ht.shape) == 2:
            ht = ht.reshape(1, *ht.shape)
        assert len(ht.shape) == 3
        resolution = ht.shape[-1]
        batch_size = ht.shape[0]
        sym_ht = np.empty((batch_size, resolution + 1, resolution + 1), dtype=ht.dtype)
        sym_ht[:, 0:-1, 0:-1] = ht
    assert resolution % 2 == 0
    sym_ht[:, -1, :] = sym_ht[:, 0]  # last row is the first row
    sym_ht[:, :, -1] = sym_ht[:, :, 0]  # last col is the first col
    sym_ht[:, -1, -1] = sym_ht[:, 0, 0]
    if len(sym_ht) == 1:
        sym_ht = sym_ht[0]
    return sym_ht


def symmetrize_ht_torch(ht):
    if ht.ndim == 2:
        ht = ht[np.newaxis, ...]
    assert ht.ndim == 3
    n = ht.shape[0]

    D = ht.shape[-1]
    sym_ht = torch.empty((n, D + 1, D + 1), dtype=ht.dtype, device=ht.device)
    sym_ht[:, 0:-1, 0:-1] = ht

    assert D % 2 == 0
    sym_ht[:, -1, :] = sym_ht[:, 0, :]  # last row is the first row
    sym_ht[:, :, -1] = sym_ht[:, :, 0]  # last col is the first col
    sym_ht[:, -1, -1] = sym_ht[:, 0, 0]  # last corner is first corner

    if n == 1:
        sym_ht = sym_ht[0, ...]

    return sym_ht