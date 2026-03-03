import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from scipy.interpolate import RegularGridInterpolator
from dataset.lut3d import Compose, RandomNoise3D_numpy, RandomIntensityShift_numpy, RandomGamma_numpy, SmoothColorWarp_numpy
import random

def resize_ratio(image, min_length):
    ratio = min(image.width, image.height) / min_length
    image = image.resize((int(image.width / ratio), int(image.height / ratio)))
    return image
# --- LUT application function (slightly optimized) ---
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
    lut = lut[..., ::-1]  

    grid = np.linspace(0, 1, size)
    interp_func = RegularGridInterpolator((grid, grid, grid), lut, bounds_error=False, fill_value=None)

    img_arr = np.asarray(image).astype(np.float32) / 255.0
    flat_pixels = img_arr.reshape(-1, 3)
    mapped_pixels = interp_func(flat_pixels)
    mapped_img = np.clip(mapped_pixels.reshape(img_arr.shape), 0, 1)

    return Image.fromarray((mapped_img * 255).astype(np.uint8))

# --- Dataset Class ---
class ImageLUTDataset(Dataset):
    """
    Dataset class for image-LUT pairs.
    """
    def __init__(self, image_dir, lut_dir, image_exts=(".jpg", ".png", ".jpeg"), lut_exts=(".npy"), return_lut_tensor=False, augment=False):
        self.images = self._get_files(image_dir, image_exts)
        self.luts = self._get_files(lut_dir, lut_exts)
        if not self.images:
            raise RuntimeError(f"No images found in {image_dir}")
        if not self.luts:
            raise RuntimeError(f"No LUTs found in {lut_dir}")
        self.return_lut_tensor = return_lut_tensor
        print(f'ImageLUTDataset images: {len(self.images)}')
        print(f'ImageLUTDataset luts: {len(self.luts)}')
        print(f'ImageLUTDataset __len__: {self.__len__()}')
        if self.return_lut_tensor:
            print("Activating normalize_to_torch process of lut data. This is expected when sending luts to vq_models.")
        if augment:
            print('ImageLUTDataset augment: True')
            self.transform =Compose([
                RandomNoise3D_numpy(sigma=0.05),
                RandomIntensityShift_numpy(brightness=0.1, contrast=0.1),
                RandomGamma_numpy(gamma_range=(0.8, 1.2)),
                SmoothColorWarp_numpy(strength=0.05),
            ])
        else:
            self.transform = None

    def _get_files(self, root, exts):
        files = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(exts):
                    files.append(os.path.join(dirpath, f))
        return sorted(files)

    def __len__(self):
        return len(self.images) * len(self.luts)

    def __getitem__(self, idx):
        img_idx = idx % len(self.images)
        lut_idx = idx % len(self.luts)

        img_path = self.images[img_idx]
        lut_path = self.luts[lut_idx]

        image = Image.open(img_path).convert("RGB")

        lut = np.load(lut_path)
        if lut.ndim != 4:
            raise ValueError(f"Expected 4D array, got shape {lut.shape} in {lut_path}")
        # Convert to (C, D, H, W) with C=3
        if lut.shape[-1] == 3:
            lut = np.transpose(lut, (3, 0, 1, 2))
        elif lut.shape[0] == 3:
            pass
        else:
            raise ValueError(f"Expected channel dim size 3, got shape {lut.shape}")
        x = lut.astype(np.float32)
        if self.transform:
            x = self.transform(x)
            x = x.astype(np.float32)
        x = np.clip(x, 0.0, 1.0)

        if self.return_lut_tensor:  
            x_torch = torch.from_numpy(x)
        else:
            x_torch = None

        toned_image = apply_lut(image, x)

        return image, toned_image, img_path, lut_path, x_torch


class ImageLUTDatasetX(Dataset):
    """
    Dataset class for image-LUT pairs.
    """
    def __init__(self, image_dir, lut_dir, image_exts=(".jpg", ".png", ".jpeg"), lut_exts=(".npy"), return_lut_tensor=False, augment=True):
        self.images = self._get_files(image_dir, image_exts)[:512]
        self.luts = self._get_files(lut_dir, lut_exts)
        if not self.images:
            raise RuntimeError(f"No images found in {image_dir}")
        if not self.luts:
            raise RuntimeError(f"No LUTs found in {lut_dir}")
        self.return_lut_tensor = return_lut_tensor
        print(f'ImageLUTDataset images: {len(self.images)}')
        print(f'ImageLUTDataset luts: {len(self.luts)}')
        print(f'ImageLUTDataset __len__: {self.__len__()}')
        if self.return_lut_tensor:
            print("Activating normalize_to_torch process of lut data. This is expected when sending luts to vq_models.")
        if augment:
            print('ImageLUTDataset augment: True')
            self.transform =Compose([
                RandomNoise3D_numpy(sigma=0.05),
                RandomIntensityShift_numpy(brightness=0.1, contrast=0.1),
                RandomGamma_numpy(gamma_range=(0.8, 1.2)),
                SmoothColorWarp_numpy(strength=0.05),
            ])
        else:
            self.transform = None

    def _get_files(self, root, exts):
        files = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(exts):
                    files.append(os.path.join(dirpath, f))
        return sorted(files)

    def __len__(self):
        return len(self.images) * len(self.luts)

    def __getitem__(self, idx):
        img_idx = idx % len(self.images)
        lut_idx = idx // len(self.luts)

        img_path = self.images[img_idx]
        lut_path = self.luts[lut_idx]

        image = Image.open(img_path).convert("RGB")
        lut = np.load(lut_path)
        if (img_idx + lut_idx) %3==0:
            img_idx_new = random.randint(0, len(self.images)-1)
            new_image_path = self.images[img_idx_new]
            new_image = Image.open(new_image_path).convert("RGB")

            toned_image = apply_lut(new_image, lut)
        else:
            toned_image = apply_lut(image, lut)
            img_idx_new = img_idx

        result_idx = f'{img_idx}_{img_idx_new}'
        if self.return_lut_tensor:
            if lut.ndim != 4:
                raise ValueError(f"Expected 4D array, got shape {lut.shape} in {lut_path}")
            # Convert to (C, D, H, W) with C=3
            if lut.shape[-1] == 3:
                lut = np.transpose(lut, (3, 0, 1, 2))
            elif lut.shape[0] == 3:
                pass
            else:
                raise ValueError(f"Expected channel dim size 3, got shape {lut.shape}")
            x = lut.astype(np.float32)
            if self.transform:
                x = self.transform(x)
                x = x.astype(np.float32)
            x = np.clip(x, 0.0, 1.0)
            x = torch.from_numpy(x)
        else:
            x = None

        return image, toned_image, result_idx, lut_path, x


class ImageLUTDataset_for_RL(Dataset):
    def __init__(self, image_dir, lut_dir, image_exts=(".jpg", ".png", ".jpeg"), lut_exts=(".npy"), augment=True, shuffle=False, resize=True, return_lut=False):
        self.images = self._get_files(image_dir, image_exts)
        self.luts = self._get_files(lut_dir, lut_exts)
        if not self.images:
            raise RuntimeError(f"No images found in {image_dir}")
        if not self.luts:
            raise RuntimeError(f"No LUTs found in {lut_dir}")

        # ---- shuffle logic ----
        self.indices = list(range(len(self)))
        if shuffle:
            random.shuffle(self.indices)

        print(f'ImageLUTDataset images: {len(self.images)}')
        print(f'ImageLUTDataset luts: {len(self.luts)}')
        print(f'ImageLUTDataset total samples: {len(self)}')

        if augment:
            print('ImageLUTDataset augment: True')
            self.augment = True
            self.transform = Compose([
                RandomNoise3D_numpy(sigma=0.05),
                RandomIntensityShift_numpy(brightness=0.2, contrast=0.2),
                RandomGamma_numpy(gamma_range=(0.9, 1.1)),
                SmoothColorWarp_numpy(strength=0.05),
            ])
        else:
            self.augment = False
            self.transform = None

        self.resize = resize
        self.return_lut = return_lut

    def _get_files(self, root, exts):
        files = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(exts):
                    files.append(os.path.join(dirpath, f))
        return sorted(files)

    def __len__(self):
        return len(self.images) * len(self.luts)

    def shuffle(self):
        """Call this at the start of each epoch if you want new order."""
        random.shuffle(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]  # map through shuffled index
        img_idx = idx % len(self.images)
        lut_idx = idx // len(self.images)

        img_path = self.images[img_idx]
        lut_path = self.luts[lut_idx]
        lut_id = lut_path.split('/')[-1].split('class_')[-1].split('.npy')[0]

        image = Image.open(img_path).convert("RGB")
        if self.resize:
            image = resize_ratio(image, 448)

        if self.augment:
            lut = np.load(lut_path)
            lut = lut.astype(np.float32)
            if lut.shape[-1] == 3:
                lut = np.transpose(lut, (3, 0, 1, 2))
            elif lut.shape[0] == 3:
                pass
            else:
                raise ValueError(f"Expected channel dim size 3, got shape {lut.shape}")
            lut = self.transform(lut)
            lut = np.clip(lut, 0.0, 1.0)
        else:
            lut = np.load(lut_path).astype(np.float32)

        img_idx_new = random.randint(0, len(self.images)-1)
        new_image_path = self.images[img_idx_new]
        new_image = Image.open(new_image_path).convert("RGB")
        if self.resize:
            new_image = resize_ratio(new_image, 448)

        ref_image = apply_lut(new_image, lut)
        ori_image = image
        gt_image = apply_lut(image, lut)

        result_idx = f'{img_idx}_{img_idx_new}_{lut_id}'

        if self.return_lut:
            return ref_image, ori_image, gt_image, result_idx, lut

        return ref_image, ori_image, gt_image, result_idx
    

