import os
import numpy as np
import torch
from torch.utils.data import Dataset

from scipy.interpolate import RegularGridInterpolator
from dataset.augment import RandomNoise3D_numpy, RandomIntensityShift_numpy, RandomGamma_numpy, SmoothColorWarp_numpy, Compose
from dataset.qwen_data import get_path

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
        lut1, lut2: 3D LUTs, shape (32, 32, 32, 3) or (3, 32, 32, 32)
        目前, 如果bs>1, 只计算第一个LUT
        standard_image: PIL.Image 对象
        method: "76", "94", "2000" 三种 ΔE 计算方式

    返回:
        平均 ΔE 
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

def apply_lut_rev(image, lut):
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

import torch
import torch.nn.functional as F

def apply_lut_batch(images, luts):
    """
    Apply a batch of 3D LUTs to a batch of images.

    Args:
        images: (N, 3, H, W) float tensor in [0,1]
        luts:   (M, S, S, S, 3) float tensor in [0,1]

    Returns:
        Tensor (N*M, 3, H, W) of LUT-applied images
    """
    N, C, H, W = images.shape
    M, S, _, _, _ = luts.shape
    device = images.device

    # Normalize pixel coords into LUT index space [0, S-1]
    coords = images.permute(0, 2, 3, 1)  # (N,H,W,3)
    coords = coords * (S - 1)

    # Scale coords into [-1,1] for grid_sample
    coords = (2.0 * coords / (S - 1)) - 1.0
    grid = coords.view(N, H, W, 3).unsqueeze(1)  # (N,1,H,W,3)

    outputs = []
    for lut in luts:
        # (1, 3, S, S, S) for grid_sample
        lut_tex = lut.permute(3, 0, 1, 2).unsqueeze(0).to(device)

        # grid_sample with 5D input (3D volume lookup)
        mapped = F.grid_sample(
            lut_tex, grid, align_corners=True, mode="bilinear"
        )  # (1,3,N,H,W) → wrong, actually (N,3,1,H,W)

        mapped = mapped.squeeze(2)  # remove depth dim (N,3,H,W)
        outputs.append(mapped)
    
    del lut_tex

    return torch.cat(outputs, dim=0)  # (N*M, 3, H, W)

def generate_standard_image(size: int = 512) -> Image.Image:
    """
    Generate a smooth test image that samples many RGB colors
    using sinusoidal projections for broad coverage.
    """
    xs = np.linspace(0.0, 2*np.pi, size, dtype=np.float32)
    ys = np.linspace(0.0, 2*np.pi, size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    rr = (np.sin(xx) * 0.5 + 0.5)
    gg = (np.sin(yy) * 0.5 + 0.5)
    bb = (np.sin(xx + yy) * 0.5 + 0.5)

    img = np.stack([rr, gg, bb], axis=2)
    img_uint8 = (img * 255).astype(np.uint8)
    return Image.fromarray(img_uint8)


def blend_luts_log(lut_a, lut_b, strength, eps=1e-6):
    """
    Blend between two LUTs in log space (per-channel).

    Args:
        lut_a, lut_b: numpy arrays of shape (size, size, size, 3)
        strength: float, interpolation factor
                  0 = lut_a, 1 = lut_b, >1 = extrapolation toward lut_b
        eps: small value to avoid log(0)

    Returns:
        blended LUT of same shape
    """
    if lut_a.shape != lut_b.shape:
        raise ValueError("LUTs must have the same shape")

    # Safe log
    log_a = np.log(np.clip(lut_a, eps, 1))
    log_b = np.log(np.clip(lut_b, eps, 1))

    # Linear blend in log space
    log_blend = (1 - strength) * log_a + strength * log_b

    # Back to linear space
    lut_blend = np.exp(log_blend)
    return np.clip(lut_blend, 0, 1)

def compose_luts(lut1, lut2):
    """
    Compose two 3D LUTs: apply lut1, then lut2.

    Args:
        lut1, lut2: numpy arrays of shape (size, size, size, 3)

    Returns:
        composed LUT of the same shape
    """
    if lut1.shape[0] == 3:
        lut1 = lut1.transpose(1, 2, 3, 0)
    if lut2.shape[0] == 3:
        lut2 = lut2.transpose(1, 2, 3, 0)
    if lut1.shape != lut2.shape:
        raise ValueError("lut1 and lut2 must have the same shape.")
    
    lut1_bgr = lut1[..., ::-1]
    lut2_bgr = lut2[..., ::-1]

    size = lut1.shape[0]
    # coordinate grid in domain [0,1]
    grid_axes = np.linspace(0.0, 1.0, size)

    # build interpolators for both LUTs (safer than relying on raw array layout)
    interp1 = RegularGridInterpolator((grid_axes, grid_axes, grid_axes),
                                     lut1_bgr, bounds_error=False, fill_value=None)
    interp2 = RegularGridInterpolator((grid_axes, grid_axes, grid_axes),
                                     lut2_bgr, bounds_error=False, fill_value=None)

    # evaluate on the canonical grid points
    pts = np.stack(np.meshgrid(grid_axes, grid_axes, grid_axes, indexing='ij'), -1).reshape(-1, 3)

    # apply lut1 (interpolated) to canonical grid
    mid = interp1(pts)            # shape (N^3, 3)

    # apply lut2 to the result of lut1
    composed = interp2(mid)

    # final clipping to valid range and reshape back to LUT grid
    composed = np.clip(composed, 0.0, 1.0)
    return composed.reshape(size, size, size, 3)[..., ::-1]

def scale_lut(lut, strength, eps=1e-6):
    """
    Interpolate between identity transform and given LUT.

    Args:
        lut: numpy array of shape (size, size, size, 3)
        strength: float, interpolation factor
                  0 = no transform, 1 = full LUT, >1 = extrapolation

    Returns:
        interpolated LUT of same shape
    """
    if lut.shape[0] == 3:
        lut = lut.transpose(1, 2, 3, 0)
    size = lut.shape[0]
    lut = lut[..., ::-1]
    # Identity LUT grid
    x = np.linspace(0, 1, size)
    identity = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)  # (size,size,size,3)
    
    log_a = np.log(np.clip(lut, eps, 1))
    log_b = np.log(np.clip(identity, eps, 1))

    # Linear blend in log space
    log_blend = strength * log_a + (1 - strength) * log_b

    # Back to linear space
    lut_blend = np.exp(log_blend)
    return np.clip(lut_blend, 0, 1)[..., ::-1]

from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

def invert_lut(lut):
    """
    Approximate inverse of a 3D LUT.

    Args:
        lut: numpy array of shape (N, N, N, 3), forward LUT in [0,1]

    Returns:
        inv_lut: numpy array of shape (N, N, N, 3), approximate inverse
    """
    if lut.shape[0] == 3:
        lut = lut.transpose(1, 2, 3, 0)
    size = lut.shape[0]
    lut = lut[..., ::-1]
    x = np.linspace(0, 1, size)
    grid = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)  # input coords
    mapped = lut.reshape(-1, 3)  # output coords

    # Build interpolator: output -> input
    # Use np.nan for fill_value so we can detect out-of-bounds
    interp = LinearNDInterpolator(mapped, grid, fill_value=np.nan)

    # Evaluate inverse mapping on the regular output grid
    inv_grid = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    inv_mapped = interp(inv_grid)

    # Handle points outside convex hull by nearest-neighbor fallback
    mask = np.isnan(inv_mapped).any(axis=1)
    if np.any(mask):
        tree = cKDTree(mapped)
        _, idx = tree.query(inv_grid[mask])
        inv_mapped[mask] = grid[idx]

    # Reshape back to LUT
    return np.clip(inv_mapped.reshape(size, size, size, 3), 0, 1)[..., ::-1]


def apply_lut_mask(image, lut, mask=None):
    """
    Apply a 3D LUT to an image, optionally within a masked region.

    Args:
        image: PIL Image object (RGB)
        lut: 3D numpy array or torch.Tensor of shape (size, size, size, 3)
        mask: optional 2D numpy array (H, W), values in [0, 1].
              Controls where and how strongly LUT is applied.

    Returns:
        PIL Image object with LUT selectively applied.
    """
    # Convert LUT if torch.Tensor
    if isinstance(lut, torch.Tensor):
        lut = lut.cpu().numpy()

    # Ensure correct channel order
    if lut.shape[0] == 3:  # (3, size, size, size)
        lut = lut.transpose(1, 2, 3, 0)

    size = lut.shape[0]
    lut = lut[..., ::-1]  # Convert RGB → BGR (if needed)

    # Create interpolation grid
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    z = np.linspace(0, 1, size)
    interp_func = RegularGridInterpolator((x, y, z), lut, bounds_error=False, fill_value=None)

    # Convert image to normalized float array
    img_arr = np.asarray(image).astype(np.float32) / 255.0
    H, W = img_arr.shape[:2]

    # Apply LUT globally
    flat_pixels = img_arr.reshape(-1, 3)
    mapped_pixels = interp_func(flat_pixels)
    mapped_img = np.clip(mapped_pixels.reshape(img_arr.shape), 0, 1)

    # If no mask → return full LUT image
    if mask is None:
        return Image.fromarray((mapped_img * 255).astype(np.uint8))

    # Ensure mask shape matches image
    if mask.shape != (H, W):
        raise ValueError(f"Mask shape {mask.shape} must match image size {(H, W)}")

    # Blend LUT and original image using mask
    # mask[..., None] → broadcast mask to RGB channels
    # operate at log level
    log_mapped_img = np.log(np.clip(mapped_img, 1e-6, 1))
    log_img_arr = np.log(np.clip(img_arr, 1e-6, 1))
    
    blended = np.exp(log_mapped_img * mask[..., None] + log_img_arr * (1 - mask[..., None]))
    blended = np.clip(blended, 0, 1)

    return Image.fromarray((blended * 255).astype(np.uint8))

import lpips
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr

# --- Helper: Convert PIL to normalized tensor ---
_lpips_model_cache = {}

def get_lpips_model(net_type='alex'):
    """Load and cache the LPIPS model (only once)."""
    if net_type not in _lpips_model_cache:
        print(f"Loading LPIPS model for {net_type}...")
        _lpips_model_cache[net_type] = lpips.LPIPS(net=net_type)
    return _lpips_model_cache[net_type]


# --- Helper: Convert PIL to normalized tensor ---
def pil_to_tensor(img):
    transform = transforms.ToTensor()  # [0,1]
    return transform(img).unsqueeze(0)  # Add batch dimension


# --- 1️⃣ LPIPS calculation ---
def calculate_lpips(img1: Image.Image, img2: Image.Image, net_type='alex') -> float:
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity) between two PIL images.

    Args:
        img1, img2: PIL.Image
        net_type: 'alex', 'vgg', or 'squeeze' (default: 'alex')

    Returns:
        float: LPIPS score (lower = more similar)
    """
    # Convert to tensors in [-1, 1]
    tensor1 = pil_to_tensor(img1) * 2 - 1
    tensor2 = pil_to_tensor(img2) * 2 - 1

    # Use cached LPIPS model
    model = get_lpips_model(net_type)

    with torch.no_grad():
        distance = model(tensor1, tensor2)

    return float(distance.item())


# --- 2️⃣ PSNR calculation ---
def calculate_psnr(img1: Image.Image, img2: Image.Image) -> float:
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        img1, img2: PIL.Image objects

    Returns:
        PSNR value (float) in dB — higher is more similar
    """
    # Convert to numpy arrays
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    
    # Handle size mismatch
    if arr1.shape != arr2.shape:
        raise ValueError("Input images must have the same dimensions")

    return psnr(arr1, arr2, data_range=255.0)
