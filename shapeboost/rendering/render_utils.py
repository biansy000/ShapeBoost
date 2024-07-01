import torch
import numpy as np
import random
import math
import torchvision.transforms as T


def augment_light_t(batch_size, device, loc_r_range=(0.05, 3.0)):
    """
    Samples batch of random point light locations.
    Azimuth/elevation is uniformly sampled over unit sphere.
        This is done by uniform sampling a location on the unit
        sphere surface via normalised Gaussian random vector.
    Distance (r) is uniformly sampled in r_range.
    :param loc_r_range: light distance range
    :return: light_t: (B, 3)
    """
    direction = torch.randn(batch_size, 3, device=device)
    direction = direction / torch.norm(direction, dim=-1)

    l, h = loc_r_range
    r = (h - l) * torch.rand(batch_size, device=device) + l

    light_t = direction * r
    return light_t


def augment_light_colour(batch_size, device,
                         ambient_intensity_range=(0.2, 0.8),
                         diffuse_intensity_range=(0.2, 0.8),
                         specular_intensity_range=(0.2, 0.8)):
    """
    Samples batch of random light INTENSITIES (not colours because
    I am forcing RGB components to be equal i.e. white lights).
    :param ambient_intensity_range: ambient component intensity range
    :param diffuse_intensity_range: diffuse component intensity range
    :param specular_intensity_range: specular component intensity range
    :return:
    """

    l, h = ambient_intensity_range
    ambient = (h - l) * torch.rand(batch_size, device=device) + l
    ambient = ambient[:, None].expand(-1, 3)

    l, h = diffuse_intensity_range
    diffuse = (h - l) * torch.rand(batch_size, device=device) + l
    diffuse = diffuse[:, None].expand(-1, 3)

    l, h = specular_intensity_range
    specular = (h - l) * torch.rand(batch_size, device=device) + l
    specular = specular[:, None].expand(-1, 3)

    return ambient, diffuse, specular


def augment_light(batch_size,
                  device,
                  rgb_augment_config):
    light_t = augment_light_t(batch_size=batch_size,
                              device=device,
                              loc_r_range=rgb_augment_config.LIGHT_LOC_RANGE)
    ambient, diffuse, specular = augment_light_colour(batch_size=batch_size,
                                                      device=device,
                                                      ambient_intensity_range=rgb_augment_config.LIGHT_AMBIENT_RANGE,
                                                      diffuse_intensity_range=rgb_augment_config.LIGHT_DIFFUSE_RANGE,
                                                      specular_intensity_range=rgb_augment_config.LIGHT_SPECULAR_RANGE)
    lights_settings = {'location': light_t,
                       'ambient_color': ambient,
                       'diffuse_color': diffuse,
                       'specular_color': specular}
    return lights_settings



def batch_add_rgb_background(backgrounds,
                             rgb,
                             seg):
    """
    :param backgrounds: (bs, 3, wh, wh)
    :param rgb: (bs, 3, wh, wh)
    :param iuv: (bs, wh, wh)
    :return: rgb_with_background: (bs, 3, wh, wh)
    """
    background_pixels = seg[:, None, :, :] == 0  # Body pixels are > 0 and out of frame pixels are -1
    rgb_with_background = rgb * (torch.logical_not(background_pixels)) + backgrounds * background_pixels
    return rgb_with_background


def inp_img_process(inp, rand_occlusion=True):
    # inp: batch x 3 x 256 x 256
    batch_size = inp.shape[0]

    if rand_occlusion:
        area_min = 0.0
        area_max = 0.5
        synth_area = (np.random.rand(batch_size) * (area_max - area_min) + area_min) * 256 * 256

        ratio_min = 0.3
        ratio_max = 1 / 0.3
        synth_ratio = (np.random.rand(batch_size) * (ratio_max - ratio_min) + ratio_min)

        synth_h = np.sqrt(synth_area * synth_ratio)
        synth_w = np.sqrt(synth_area / synth_ratio)
        synth_xmin = np.random.rand(batch_size) * (256 - synth_w - 1)
        synth_ymin = np.random.rand(batch_size) * (256 - synth_h - 1)

        for i in range(batch_size):
            left = int(synth_xmin[i])
            top = int(synth_ymin[i])
            right = int(synth_xmin[i] + synth_w[i])
            down = int(synth_ymin[i] + synth_h[i])
            if left >= 0 and top >= 0 and right < 256 and down < 256 and random.random() < 0.5:
                # print('correct?', left >= 0 and top >= 0 and right < 256 and down < 256)
                inp[i, :, top:down, left:right] = torch.rand(3, down-top, right-left, device=inp.device, dtype=torch.float32)
            
            inp[i] = img_augmentation(inp[i])

    inp_img_mean = torch.tensor([0.406, 0.457, 0.480], device=inp.device).float()
    inp_img_std = torch.tensor([0.225, 0.224, 0.229], device=inp.device).float()
    
    inp = inp - inp_img_mean.reshape(1, 3, 1, 1)
    inp = inp / inp_img_std.reshape(1, 3, 1, 1)

    return inp


transforms_aug = []
transforms_aug.append(T.ColorJitter(brightness=0.5, hue=0.25, contrast=0.5, saturation=0.5))
transforms_aug.append(T.RandomGrayscale(p=0.1))
transforms_aug = T.Compose(transforms_aug)
def img_augmentation(img_torch):
    r, r1, r2 = [random.random() for _ in range(3)]
    if r < 0.5:
        size = int(96 + r * (256 - 96) * 2)
        img_torch = T.Resize(size=size)(img_torch)
        img_torch = transforms_aug(img_torch)
        img_torch = T.Resize(size=256)(img_torch)
    elif r < 0.8:
        img_torch = T.GaussianBlur(
            kernel_size=(int(3*r1)*2+3, int(3*r2)*2+3), sigma=(0.1, 3) )(img_torch)
        
        img_torch = transforms_aug(img_torch)

    return img_torch


import torch.nn as nn
import torch.nn.functional as F
class Conv2clothes(nn.Module):
    def __init__(self, max_layer_num=64, dtype=torch.float32):
        super(Conv2clothes, self).__init__()

        weight_filter = torch.ones(3, 3, dtype=dtype)

        self.register_buffer(
            'filter', weight_filter.reshape(1, 1, 3, 3))

        self.max_layer_num = max_layer_num
        self.dtype=dtype
    
    def forward(self, original_seg):
        # original_seg : batch x 1 x height x width
        with torch.no_grad():
            # batch_size, height, width = original_seg.shape[0], original_seg.shape[-2], original_seg.shape[-1]
            seg_inp = (original_seg < 0.01) * 1.0 # background
            seg_out = F.conv2d(seg_inp, self.filter, stride=1, padding=1)

            clothes = ((seg_out > 0.01) & (original_seg > 0.01)) * 1.0

            return clothes

