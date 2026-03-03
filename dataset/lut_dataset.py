import os
import numpy as np
import torch
from torch.utils.data import Dataset

from scipy.interpolate import RegularGridInterpolator
from dataset.augment import RandomNoise3D_numpy, RandomIntensityShift_numpy, RandomGamma_numpy, SmoothColorWarp_numpy, Compose
import random
from scipy.ndimage import gaussian_filter
from dataset.qwen_data import get_path

# self-contained file

# usage:
# dataset = LUTDataset(augment = False, return_tensor=False)
# lut = dataset[0]
# image_lut = apply_lut(image, lut)

class RandomNoise3D_tensor:
    """Add small Gaussian noise to RGB values, preserving smoothness."""
    def __init__(self, sigma=0.01, smooth=True, smooth_sigma=1.0):
        self.sigma = sigma
        self.smooth = smooth
        self.smooth_sigma = smooth_sigma
    def __call__(self, tensor):
        noise = np.random.randn(*tensor.shape) * self.sigma
        if self.smooth:
            for c in range(noise.shape[0]):  # channel-first: (C, D, H, W)
                noise[c] = gaussian_filter(noise[c], self.smooth_sigma)
        return torch.clamp(tensor + torch.from_numpy(noise), 0.0, 1.0)


class RandomIntensityShift_tensor:
    """Brightness + contrast adjustment."""
    def __init__(self, brightness=0.05, contrast=0.05):
        self.brightness = brightness
        self.contrast = contrast
    def __call__(self, tensor):
        # shift brightness
        shift = random.uniform(-self.brightness, self.brightness)
        # scale contrast
        scale = 1.0 + random.uniform(-self.contrast, self.contrast)
        mean_val = 0.5
        tensor = (tensor - mean_val) * scale + mean_val + shift
        return torch.clamp(tensor, 0.0, 1.0)

class RandomNoise3D_numpy:
    """Add smooth Gaussian noise to RGB values in LUT."""
    def __init__(self, sigma=0.01, smooth=True, smooth_sigma=1.0):
        self.sigma = sigma
        self.smooth = smooth
        self.smooth_sigma = smooth_sigma

    def __call__(self, tensor):
        noise = np.random.randn(*tensor.shape) * self.sigma
        if self.smooth:
            for c in range(noise.shape[0]):  # channel-first: (C, D, H, W)
                noise[c] = gaussian_filter(noise[c], self.smooth_sigma)
        return np.clip(tensor + noise, 0.0, 1.0)


class RandomIntensityShift_numpy:
    """Brightness + contrast adjustment (contrast centered at 0.5)."""
    def __init__(self, brightness=0.05, contrast=0.05):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, tensor):
        shift = random.uniform(-self.brightness, self.brightness)
        scale = 1.0 + random.uniform(-self.contrast, self.contrast)
        mean_val = 0.5
        tensor = (tensor - mean_val) * scale + mean_val + shift
        return np.clip(tensor, 0.0, 1.0)


class RandomGamma_numpy:
    """Apply random gamma correction to RGB values."""
    def __init__(self, gamma_range=(0.9, 1.1)):
        self.gamma_range = gamma_range

    def __call__(self, tensor):
        gamma = random.uniform(*self.gamma_range)
        return np.power(np.clip(tensor, 0.0, 1.0), gamma)


class SmoothColorWarp_numpy:
    """Apply a smooth linear color warp (near-identity 3x3 matrix)."""
    def __init__(self, strength=0.05):
        self.strength = strength

    def __call__(self, tensor):
        C, D, H, W = tensor.shape
        assert C == 3, "Expected RGB channels"

        A = np.eye(3) + np.random.randn(3, 3) * self.strength

        flat = tensor.reshape(3, -1)
        warped = A @ flat
        warped = warped.reshape(C, D, H, W)

        return np.clip(warped, 0.0, 1.0)


class Compose:
    """Compose several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
    
class LUTDataset(Dataset):
    """Loads 3D LUTs from .npy files.
    Accepts arrays shaped (32, 32, 32, 3) or (3, 32, 32, 32).
    Values are expected in [0,1].
    return tensor if return_tensor else numpy array in shape (32,32,32,3)
    """
    def __init__(self, data_dir: str = get_path("luts_3d_npy_dir", "PATH_TO_LUTS_3D_NPY_DIR"), augment: bool=False, return_tensor=True):
        self.paths = []
        for subdir, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".npy"):
                    self.paths.append(os.path.join(subdir, f))

        if len(self.paths) == 0:
            raise RuntimeError(f"No .npy files found under {data_dir}") 
        self.augment = augment
        self.return_tensor = return_tensor
        self.transform = Compose([
                RandomNoise3D_numpy(sigma=0.05),
                RandomIntensityShift_numpy(brightness=0.1, contrast=0.1),
                RandomGamma_numpy(gamma_range=(0.8, 1.2)),
                SmoothColorWarp_numpy(strength=0.05),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = np.load(self.paths[idx])
        if arr.ndim != 4:
            raise ValueError(f"Expected 4D array, got shape {arr.shape} in {self.paths[idx]}")
        # Convert to (C, D, H, W) with C=3
        if arr.shape[-1] == 3:
            arr = np.transpose(arr, (3, 0, 1, 2))
        elif arr.shape[0] == 3:
            pass
        else:
            raise ValueError(f"Expected channel dim size 3, got shape {arr.shape}")
        x = arr.astype(np.float32)
        if self.augment:
            x = self.transform(x)
            x = x.astype(np.float32)
        x = np.clip(x, 0.0, 1.0)
        if self.return_tensor:
            return torch.from_numpy(x)
        else:
            return x

from PIL import Image
import numpy as np
from skimage import color

def delta_e_between_images(img1: Image.Image, img2: Image.Image, method: str = "2000") -> float:
    """
    计算两张图像的平均 Delta E (颜色差异)

    参数:
        img1, img2: PIL.Image 对象，大小需要一致
        method: "76", "94", "2000" 三种 ΔE 计算方式

    返回:
        平均 ΔE 值
    """
    if img1.size != img2.size:
        raise ValueError("两张图片的大小必须一致")

    # 转为 numpy 数组并归一化到 [0,1]
    arr1 = np.asarray(img1.convert("RGB"), dtype=np.float32) / 255.0
    arr2 = np.asarray(img2.convert("RGB"), dtype=np.float32) / 255.0

    # 转换到 Lab 颜色空间
    lab1 = color.rgb2lab(arr1)
    lab2 = color.rgb2lab(arr2)

    # 计算 ΔE
    if method == "76":
        delta_e = color.deltaE_cie76(lab1, lab2)
    elif method == "94":
        delta_e = color.deltaE_ciede94(lab1, lab2)
    elif method == "2000":
        delta_e = color.deltaE_ciede2000(lab1, lab2)
    else:
        raise ValueError("method 必须是 '76', '94' 或 '2000'")

    return float(np.mean(delta_e))  # 返回平均差异


def delta_e_between_images_for_luts(lut1: np.ndarray, lut2: np.ndarray, standard_image, method: str = "2000") -> float:
    """
    计算两张图像的平均 Delta E (颜色差异)

    参数:
        img1, img2: PIL.Image 对象，大小需要一致
        method: "76", "94", "2000" 三种 ΔE 计算方式

    返回:
        平均 ΔE 值
    """

    if lut1.shape != lut2.shape:
        raise ValueError("LUTs的大小必须一致")
    # if you send more than one LUTS, we only compute the first to save time for now.
    if lut1.ndim > 4:
        lut1 = lut1[0]
    if lut2.ndim > 4:
        lut2 = lut2[0]
    img1 = apply_lut(standard_image, lut1)
    img2 = apply_lut(standard_image, lut2)

    # 转为 numpy 数组并归一化到 [0,1]
    arr1 = np.asarray(img1.convert("RGB"), dtype=np.float32) / 255.0
    arr2 = np.asarray(img2.convert("RGB"), dtype=np.float32) / 255.0

    # 转换到 Lab 颜色空间
    lab1 = color.rgb2lab(arr1)
    lab2 = color.rgb2lab(arr2)

    # 计算 ΔE
    if method == "76":
        delta_e = color.deltaE_cie76(lab1, lab2)
    elif method == "94":
        delta_e = color.deltaE_ciede94(lab1, lab2)
    elif method == "2000":
        delta_e = color.deltaE_ciede2000(lab1, lab2)
    else:
        raise ValueError("method 必须是 '76', '94' 或 '2000'")

    return float(np.mean(delta_e)), img1, img2  # return dE and toned images.

def apply_lut(image, lut):
    """
    Apply a 3D LUT to an image.

    Args:
        image: PIL Image object
        lut: 3D numpy array of shape (size, size, size, 3)

    Returns:
        PIL Image object with LUT applied
    """
    if isinstance(lut, torch.Tensor):
        lut = lut.cpu().numpy()
    if lut.shape[0] == 3:
        lut = lut.transpose(1, 2, 3, 0)
    size = lut.shape[0]
    lut = lut[..., ::-1] # this can only be done to np.arrays. not torch tensors.

    # convert to bgr
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    z = np.linspace(0, 1, size)
    interp_func = RegularGridInterpolator((x, y, z), lut, bounds_error=False, fill_value=None)

    img_arr = np.asarray(image).astype(np.float32) / 255.0
    flat_pixels = img_arr.reshape(-1, 3)
    mapped_pixels = interp_func(flat_pixels)
    mapped_img = np.clip(mapped_pixels.reshape(img_arr.shape), 0, 1)
    return Image.fromarray((mapped_img * 255).astype(np.uint8))
