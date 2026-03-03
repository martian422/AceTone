import os
import numpy as np
import torch
import random
from scipy.ndimage import gaussian_filter

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

def create_gradient_mask(image, mask_type="linear", **kwargs):
    """
    Create a gradient mask (linear or elliptical) for selective color tuning.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image (RGB or grayscale).
    mask_type : str
        'linear' or 'ellipse'.

    For mask_type='linear':
        angle : float (degrees, 0 = left→right, 90 = top→bottom)
        start : float (0-1, start of gradient)
        end : float (0-1, end of gradient)

    For mask_type='ellipse':
        center : (x_ratio, y_ratio)
        radius : (x_radius_ratio, y_radius_ratio)
        angle : float (degrees, 0 = horizontal, 90 = vertical)
        feather : float (softness of edge, e.g. 0.2)

    invert : bool (default False)
        If True, invert the mask.
        
    Returns
    -------
    mask : np.ndarray
        2D float mask in [0, 1], same size as the image.
    """
    # Convert PIL image to NumPy
    W, H = image.size
    y = np.linspace(0, 1, H)
    x = np.linspace(0, 1, W)
    xv, yv = np.meshgrid(x, y)

    if mask_type == "linear":
        angle = np.deg2rad(kwargs.get("angle", 90))  # degrees → radians
        start = kwargs.get("start", 0.0)
        end = kwargs.get("end", 1.0)

        # Compute gradient direction
        # Rotate coordinates around center
        cx, cy = 0.5, 0.5
        xv_rot = (xv - cx) * np.cos(angle) + (yv - cy) * np.sin(angle)
        
        # Normalize xv_rot into [0, 1]
        xv_min, xv_max = xv_rot.min(), xv_rot.max()
        grad = (xv_rot - xv_min) / (xv_max - xv_min + 1e-8)
        
        # Apply start/end limits
        mask = np.clip((grad - start) / (end - start + 1e-8), 0, 1)

    elif mask_type == "ellipse":
        cx, cy = kwargs.get("center", (0.5, 0.5))
        rx, ry = kwargs.get("radius", (0.4, 0.4))
        angle = np.deg2rad(kwargs.get("angle", 0))  # degrees → radians
        feather = kwargs.get("feather", 0.2)

        # Rotate coordinates around center
        xv_rot = (xv - cx) * np.cos(angle) + (yv - cy) * np.sin(angle)
        yv_rot = -(xv - cx) * np.sin(angle) + (yv - cy) * np.cos(angle)
        
        dist = np.sqrt(((xv_rot / rx) ** 2 + (yv_rot / ry) ** 2))
        mask = np.clip((1 - dist) / feather, 0, 1)

    else:
        raise ValueError(f"Unknown mask_type '{mask_type}'")
    
    # if get invert, then 1 - mask
    if kwargs.get("invert", False):
        mask = 1 - mask
    return mask