
import torchvision.transforms as transforms
# from torchtoolbox.transform import Cutout
import numpy as np
import random
import math
import PIL
from torch import nn
import torch
# import cv2
from PIL import ImageFilter
import time
from PIL import Image, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


def get_crop_mask(width, crop_ratio, replace=0.0):
    mask = np.zeros([width, width])
    center = np.array([width / 2, width / 2])
    for i in range(width):
        for j in range(width):
            if math.sqrt(
                    (i - int(center[0])) ** 2 + (j - int(center[1])) ** 2) < crop_ratio * width * 0.5:
                mask[i, j] = 1
    return mask


class crop(object):

    def __init__(self, img_width=128, ratio=1.0):
        self.mask = get_crop_mask(width=img_width, crop_ratio=ratio)

    def __call__(self, img, img_width=128, ratio=1.0):
        img = np.array(img)
        img = img * self.mask
        return PIL.Image.fromarray(img)





class with_frequency_domain(object):
    def __init__(self, is_add=True, is_only=False, strategy='patch', patch_size=16):
        self.is_add = is_add
        self.is_only = is_only
        self.strategy = strategy
        self.patch_size = patch_size

    def __call__(self, img):
        if self.is_add:
            # fft_real_norm= (img_fft.real - img_fft.real.mean()) / (img_fft.real.max() - img_fft.real.min())*(img.max()-img.min())
            # fft_imag_norm = (img_fft.imag - img_fft.imag.mean()) / (img_fft.imag.max()-img_fft.imag.min())*(img.max()-img.min())
            fft_real_norm, fft_imag_norm = get_frequency_domain(img, mode=self.strategy, patch_size=self.patch_size)
            img_combined = torch.cat((img, fft_real_norm, fft_imag_norm), 0)
        elif self.is_only:
            # fft_real_norm= (img_fft.real - img_fft.real.mean()) / (img_fft.real.max() - img_fft.real.min())*(img.max()-img.min())
            # fft_imag_norm = (img_fft.imag - img_fft.imag.mean()) / (img_fft.imag.max()-img_fft.imag.min())*(img.max()-img.min())
            fft_real_norm, fft_imag_norm = get_frequency_domain(img, mode=self.strategy, patch_size=self.patch_size)
            img_combined = torch.cat((fft_real_norm, fft_imag_norm), 0)
        else:
            img_combined = img
        return img_combined


# In the get_frequency_domain function, how to Parallelize the patch processing?
# def get_frequency_domain(img,mode,patch_size=14):
#     if mode=='whole':
#         img_fft = torch.fft.fft2(img)
#         fft_real_norm = (img_fft.real - img_fft.real.mean()) / (img_fft.real.max() - img_fft.real.min()) * (
#                     img.max() - img.min())
#         fft_imag_norm = (img_fft.imag - img_fft.imag.mean()) / (img_fft.imag.max() - img_fft.imag.min()) * (
#                     img.max() - img.min())
#     elif mode=='patch':
#         fft_real_norm = torch.zeros(img.shape)
#         fft_imag_norm = torch.zeros(img.shape)
#         for i in range(0,img.shape[2],patch_size):
#             for j in range(0,img.shape[1],patch_size):
#                 img_patch=img[:,i:i+patch_size,j:j+patch_size]
#                 img_fft = torch.fft.fft2(img_patch)
#                 # fft_real_norm[i:i+patch_size,j:j+patch_size] = img_fft.real
#                 # fft_imag_norm[i:i+patch_size,j:j+patch_size] = img_fft.imag
#                 scale=img_patch.max() - img_patch.min()
#                 fft_real_norm[:,i:i+patch_size,j:j+patch_size] = scale* (img_fft.real-img_fft.real.mean()) / (img_fft.real.max() - img_fft.real.min())
#                 if img_fft.imag.max() - img_fft.imag.min()!=0:
#                     fft_imag_norm[:,i:i+patch_size,j:j+patch_size] = scale*(img_fft.imag-img_fft.imag.mean()) / (img_fft.imag.max() - img_fft.imag.min())
#                 else:
#                     fft_imag_norm[:,i:i+patch_size,j:j+patch_size] = scale*(img_fft.imag-img_fft.imag.mean())
#         # fft_real_norm = (fft_real_norm - fft_real_norm.mean()) / (fft_real_norm.max() - fft_real_norm.min()) * (
#         #             img.max() - img.min())
#         # fft_imag_norm = (fft_imag_norm - fft_imag_norm.mean()) / (fft_imag_norm.max() - fft_imag_norm.min()) * (
#         #             img.max() - img.min())
#     else:
#         raise ValueError('frequency domain transform: mode error')
#     return fft_real_norm,fft_imag_norm


def get_frequency_domain(img, mode, patch_size=14):
    if mode == 'whole':
        img_fft = torch.fft.fft2(img)
        fft_real_norm = (img_fft.real - img_fft.real.mean()) / (img_fft.real.max() - img_fft.real.min()) * (
                img.max() - img.min())
        fft_imag_norm = (img_fft.imag - img_fft.imag.mean()) / (img_fft.imag.max() - img_fft.imag.min()) * (
                img.max() - img.min())

    elif mode == 'patch':
        # Get the dimensions of the image
        batch_size, height, width = img.shape

        # Number of patches along the height and width
        num_patches_height = (height + patch_size - 1) // patch_size
        num_patches_width = (width + patch_size - 1) // patch_size

        # Pad the image if necessary to fit the patch size exactly
        pad_height = patch_size * num_patches_height - height
        pad_width = patch_size * num_patches_width - width
        img_padded = torch.nn.functional.pad(img, (0, pad_width, 0, pad_height), mode='constant', value=0)

        # Reshape and permute to create a batch of patches
        img_patches = img_padded.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        img_patches = img_patches.contiguous().view(batch_size, -1, patch_size, patch_size)

        # Compute FFT on all patches at once
        img_fft = torch.fft.fft2(img_patches)

        # Normalize the real and imaginary parts separately
        # Replace the tuple dimension with sequential operations
        scale = img_patches.max(dim=-1)[0].max(dim=-1)[0] - img_patches.min(dim=-1)[0].min(dim=-1)[0]
        scale = scale.unsqueeze(-1).unsqueeze(-1)

        mean_real = img_fft.real.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        min_real = img_fft.real.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_real = img_fft.real.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        mean_imag = img_fft.imag.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        min_imag = img_fft.imag.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_imag = img_fft.imag.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        fft_real_norm_patches = (img_fft.real - mean_real) / (max_real - min_real) * scale
        fft_imag_norm_patches = (img_fft.imag - mean_imag) / (max_imag - min_imag) * scale

        # Handle cases where the denominator is 0
        fft_imag_norm_patches[torch.isnan(fft_imag_norm_patches)] = 0

        # Reshape normalized patches to the original image layout
        fft_real_norm = torch.zeros_like(img_padded)
        fft_imag_norm = torch.zeros_like(img_padded)

        fft_real_norm_patches = fft_real_norm_patches.view(batch_size, num_patches_height, num_patches_width,
                                                           patch_size, patch_size)
        fft_imag_norm_patches = fft_imag_norm_patches.view(batch_size, num_patches_height, num_patches_width,
                                                           patch_size, patch_size)

        for i in range(num_patches_height):
            for j in range(num_patches_width):
                fft_real_norm[:, i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size] = fft_real_norm_patches[:, i, j]
                fft_imag_norm[:, i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size] = fft_imag_norm_patches[:, i, j]

        # Remove the padding if there was any
        fft_real_norm = fft_real_norm[:, :height, :width]
        fft_imag_norm = fft_imag_norm[:, :height, :width]

    else:
        raise ValueError('frequency domain transform: mode error')

    return fft_real_norm, fft_imag_norm


def filter(img_width, D0, N=2, type='lp', filter='butterworth'):
    '''
    频域滤波器
    Args:
        img: 灰度图片
        D0: 截止频率
        N: butterworth的阶数(默认使用二阶)
        type: lp-低通 hp-高通
        filter:butterworth、ideal、Gaussian即巴特沃斯、理想、高斯滤波器
    Returns:
        imgback：滤波后的图像
    '''

    rows = img_width
    cols = img_width
    crow, ccol = int(rows / 2), int(cols / 2)  # 计算频谱中心
    mask = np.zeros((rows, cols))  # 生成rows行cols列的二维矩阵
    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)  # 计算D(u,v)
            if (filter.lower() == 'butterworth'):  # 巴特沃斯滤波器
                if (type == 'lp'):
                    mask[i, j] = 1 / (1 + (D / D0) ** (2 * N))
                elif (type == 'hp'):
                    mask[i, j] = 1 / (1 + (D0 / D) ** (2 * N))
                else:
                    assert ('type error')
            elif (filter.lower() == 'ideal'):  # 理想滤波器
                if (type == 'lp'):
                    if (D <= D0):
                        mask[i, j] = 1
                elif (type == 'hp'):
                    if (D > D0):
                        mask[i, j] = 1
                else:
                    assert ('type error')
            elif (filter.lower() == 'gaussian'):  # 高斯滤波器
                if (type == 'lp'):
                    mask[i, j] = np.exp(-(D * D) / (2 * D0 * D0))
                elif (type == 'hp'):
                    mask[i, j] = (1 - np.exp(-(D * D) / (2 * D0 * D0)))
                else:
                    assert ('type error')
    return mask


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, snr_scale=[0.1, 1]):
        self.mean = mean
        self.max_snr = snr_scale[1]
        self.min_snr = snr_scale[0]

    def __call__(self, img):
        # img = np.array(img)
        # snr=random.uniform(self.min_snr,self.max_snr)
        # N = np.random.normal(self.mean, (img.var() / snr) ** 0.5, img.shape)
        # noised_img = N + img
        # if img.dtype=='uint8':
        #     noised_img[noised_img > 255] = 255                       # 避免有值超过255而反转
        #     noised_img[noised_img < 0] = 0                       # 避免有值超过255而反转
        #     noised_img=noised_img.astype('uint8')
        # noised_img = PIL.Image.fromarray(noised_img)

        sigma = random.uniform(self.min_snr, self.max_snr)
        noised_img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return noised_img


class AddSaltPepperNoise(object):

    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        img = np.array(img)
        noised_img = img.copy()
        max = np.max(img)
        min = np.min(img)
        h, w = img.shape
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(h, w), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask

        noised_img[mask == 0] = min  # 椒
        noised_img[mask == 1] = max  # 盐
        if img.dtype == 'uint8':
            noised_img[noised_img > 255] = 255  # 避免有值超过255而反转
            noised_img[noised_img < 0] = 0  # 避免有值超过255而反转
            noised_img = noised_img.astype('uint8')
        noised_img = PIL.Image.fromarray(noised_img)  # numpy转图片
        return noised_img


class rondom_pixel_lost(object):

    def __init__(self, p=0.5, ratio=(0.4, 1 / 0.4)):

        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")
        self.p = p
        self.ratio = ratio

    def __call__(self, img):
        img = np.array(img)
        if random.random() < self.p:
            img_h, img_w = img.shape
            lost_pixel_num = int(img.size * self.ratio)
            mask = np.concatenate((np.zeros(lost_pixel_num), np.ones(img.size - lost_pixel_num)), axis=0)
            np.random.shuffle(mask)
            mask = mask.reshape((img_w, img_h))
            img = img * mask
        return PIL.Image.fromarray(img)


class my_normalize(object):
    def __call__(self, mrcdata):
        # time0=time.time()
        min_val, max_val = mrcdata.getextrema()
        if max_val - min_val != 0:
            mrcdata.point(lambda i: (i - min_val) / (max_val - min_val))
        # mrcdata=transforms.Normalize(mean=[min_val], std=[max_val-min_val])(mrcdata)
        # transform1=time.time()-time0
        #
        # time1=time.time()
        # mrcdata_np = (np.array(mrcdata))
        # if np.max(mrcdata) - np.min(mrcdata) != 0:
        #     mrcdata = (mrcdata - np.min(mrcdata)) / (np.max(mrcdata) - np.min(mrcdata))
        # mrcdata = mrcdata.astype(np.float32)
        # tranform2=time.time()-time1
        return mrcdata


class random_brightness(object):
    def __init__(self, scale=0.5):
        self.scale = scale

    def __call__(self, mrcdata):
        # mrcdata = np.array(mrcdata)
        # mrcdata_tensor = torch.tensor(mrcdata).unsqueeze(0)
        aug = transforms.ColorJitter(brightness=self.scale)(mrcdata)
        # aug_np=aug.numpy()
        return aug


class random_contrast(object):
    def __init__(self, scale=0.5):
        self.scale = scale

    def __call__(self, mrcdata):
        # mrcdata = np.array(mrcdata)
        # mrcdata_tensor = torch.tensor(mrcdata).unsqueeze(0)
        aug = transforms.ColorJitter(contrast=self.scale)(mrcdata)
        # aug_np=aug.numpy()
        # return transforms.ToPILImage()(aug)
        return aug


def to_int8(mrcdata):
    mrcdata = np.array(mrcdata)
    # a=np.max(mrcdata)
    # b=np.min(mrcdata)
    if np.max(mrcdata) - np.min(mrcdata) != 0:
        mrcdata = (mrcdata - np.min(mrcdata)) / ((np.max(mrcdata) - np.min(mrcdata)))
        mrcdata = (mrcdata * 255).astype(np.uint8)
    else:
        mrcdata = mrcdata.astype(np.uint8)

    return PIL.Image.fromarray(mrcdata)


# def my_normalize(mrcdata):
#     mrcdata = (mrcdata - torch.min(mrcdata)) / ((torch.max(mrcdata) - torch.min(mrcdata)))
#     return mrcdata

def get_train_transformations(p, mean_std=None, patch_size=16):
    my_transform = transforms.Compose([])
    if p['augmentation_strategy'] == 'user-defined':
        if 'random_flip' in p:
            my_transform = transforms.Compose(
                my_transform.transforms + [transforms.RandomHorizontalFlip(p['random_flip'])] + [
                    transforms.RandomVerticalFlip(p['random_flip'])])

        if 'random_pixel_lost' in p:
            my_transform = transforms.Compose(
                my_transform.transforms + [rondom_pixel_lost(p['random_pixel_lost']['p'],
                                                             ratio=p['random_pixel_lost'][
                                                                 'ratio'])])
        if 'random_resized_crop' in p:
            my_transform = transforms.Compose(
                my_transform.transforms + [RandomApply(
                    transforms.RandomResizedCrop(**p['random_resized_crop']['setting']),
                    p=p['random_resized_crop']['p']
                ), ])

        if 'center_crop' in p:
            my_transform = transforms.Compose(
                my_transform.transforms + [RandomApply(
                    transforms.CenterCrop(**p['random_resized_crop']['setting']),
                    p=p['random_resized_crop']['p']
                ), ])

        # if 'cutout_kwargs' in p:
        #     pass
        # my_transform = transforms.Compose(
        #     my_transform.transforms + [Cutout(p['cutout_kwargs']['p'],
        #                                       value=p['cutout_kwargs']['value'])])

        # if 'random_rotate' in p:
        #     my_transform = transforms.Compose(
        #         my_transform.transforms + [transforms.RandomRotation(p['random_rotate'])])

        if 'gaussian_blur' in p:
            # my_transform = transforms.Compose(
            #     my_transform.transforms + [RandomApply(
            #         transforms.GaussianBlur((3, 3), (1.0, 2.0)),
            #         p=p['gaussian_blur']['p']
            #     ), ])
            my_transform = transforms.Compose(
                my_transform.transforms + [
                    GaussianBlur(p=p['gaussian_blur']['p'], radius_min=p['gaussian_blur']['radius_scale'][0],
                                 radius_max=p['gaussian_blur']['radius_scale'][1])])

        if 'gaussian_noise' in p:
            my_transform = transforms.Compose(
                my_transform.transforms + [RandomApply(
                    AddGaussianNoise(snr_scale=p['gaussian_noise']['snr_scale']),
                    p=p['gaussian_noise']['p']
                ), ])

        if 'solarization' in p:
            my_transform = transforms.Compose(
                my_transform.transforms + [Solarization(p=p['solarization']['p'])])

        # if 'brightness' in p:
        #     my_transform = transforms.Compose(
        #         my_transform.transforms + [RandomApply(
        #             transforms.ColorJitter(brightness=p['brightness']['scale']),
        #             p=p['brightness']['p']
        #         ), ])
        #
        # if 'contrast' in p:
        #     my_transform = transforms.Compose(
        #         my_transform.transforms + [RandomApply(
        #             random_contrast(scale=p['contrast']['scale']),
        #             p=p['contrast']['p']
        #         ), ])

        if 'colorjitter' in p:
            my_transform = transforms.Compose(
                my_transform.transforms + [RandomApply(
                    transforms.ColorJitter(brightness=p['colorjitter']['brightness'],
                                           contrast=p['colorjitter']['contrast'],
                                           saturation=p['colorjitter']['saturation'], hue=p['colorjitter']['hue']),
                    p=p['colorjitter']['p']
                ), ])

        if 'saltpepper_noise' in p:
            my_transform = transforms.Compose(
                my_transform.transforms + [RandomApply(
                    AddSaltPepperNoise(p['saltpepper_noise']['d']),
                    p=p['saltpepper_noise']['p']
                ), ])

        my_transform = transforms.Compose(
            [transforms.RandomOrder(my_transform.transforms)])

        if 'resize' in p:
            my_transform = transforms.Compose(
                [transforms.Resize(p['resize'])] + my_transform.transforms)

        # if p['is_Normalize']:
        #     my_transform = transforms.Compose(
        #         my_transform.transforms +
        #         [my_normalize()])

        if 'crop' in p:
            my_transform = transforms.Compose(
                my_transform.transforms + [crop(p['resize'], p['crop']['r'])])



        my_transform = transforms.Compose(
            my_transform.transforms + [transforms.ToTensor()])
        if mean_std is not None:
            my_transform = transforms.Compose(
                my_transform.transforms +
                [transforms.Normalize(mean=mean_std[0], std=mean_std[1])])
        if p['frequency_domain']['strategy'] is not None:
            my_transform = transforms.Compose(my_transform.transforms + [
                with_frequency_domain(is_add=p['frequency_domain']['is_add'], is_only=p['frequency_domain']['is_only'],
                                      patch_size=patch_size, strategy=p['frequency_domain']['strategy'])])




    else:
        my_transform = transforms.TrivialAugmentWide()
        my_transform = transforms.Compose(
            [to_int8, RandomApply(transforms.RandomRotation(p['random_rotate']['angle']), p['random_rotate']['p']),
             transforms.Resize(p['resize']),
             my_transform, transforms.ToTensor()])
    if 'random_rotate' in p:
        random_rotate = RandomApply(transforms.RandomRotation(p['random_rotate']['angle']), p['random_rotate']['p'])
    else:
        random_rotate = None
    return [my_transform, get_localcrop_transformations(p, mean_std=mean_std), random_rotate]


# def get_old_model(p):

def get_val_transformations(p, patch_size=16,mean_std=None):
    my_transform = transforms.Compose([])



    if 'resize' in p:
        my_transform = transforms.Compose(
            [transforms.Resize(p['resize'])] + my_transform.transforms)

    # if p['is_Normalize']:
    #     my_transform = transforms.Compose(
    #         my_transform.transforms +
    #         [my_normalize()])

    if 'crop' in p:
        my_transform = transforms.Compose(
            my_transform.transforms + [crop(p['resize'], p['crop']['r'])])

    my_transform = transforms.Compose(
        # [to_int8] +
        my_transform.transforms + [transforms.ToTensor()])
    if mean_std is not None:
        my_transform = transforms.Compose(
            my_transform.transforms +
            [transforms.Normalize(mean=mean_std[0], std=mean_std[1])])

    if p['frequency_domain']['strategy'] is not None:
        my_transform = transforms.Compose(my_transform.transforms + [
            with_frequency_domain(is_add=p['frequency_domain']['is_add'], is_only=p['frequency_domain']['is_only'],
                                      patch_size=patch_size, strategy=p['frequency_domain']['strategy'])])

    return [my_transform, None, None]


def get_localcrop_transformations(p, mean_std=None):
    # local_crops_scale = p['local_crops']['scale']
    # my_transform = transforms.Compose([
    #     transforms.RandomRotation(p['local_crops']['random_rotate']),
    #     transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
    #     # transforms.RandomRotation([-15,15]),
    #     RandomApply(
    #         transforms.ColorJitter(brightness=p['local_crops']['colorjitter']['brightness'],
    #                                contrast=p['local_crops']['colorjitter']['contrast'],
    #                                saturation=p['local_crops']['colorjitter']['saturation'], hue=p['local_crops']['colorjitter']['hue']),
    #         p=p['local_crops']['colorjitter']['p']
    #     ),
    #     GaussianBlur(p=p['local_crops']['gaussian_blur']['p'], radius_min=p['local_crops']['gaussian_blur']['radius_scale'][0],
    #                              radius_max=p['local_crops']['gaussian_blur']['radius_scale'][1]),
    #     transforms.ToTensor(),
    # ])
    # if mean_std is not None:
    #     my_transform = transforms.Compose(
    #         my_transform.transforms +
    #         [transforms.Normalize(mean=mean_std[0], std=mean_std[1])])
    # return my_transform
    return None


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img





#
#
# if __name__ == '__main__':
#     transformers_test()
    # filter_test()
