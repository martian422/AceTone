import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

def resize(*imgs):
    ## resize all the images so that they have the same width, do not change the aspect ratio
    ## the image size shall match the settings during pretraining for best performance, but a variant is also acceptable.
    width = 512
    return [img.resize((width, int(width * img.height / img.width))) for img in imgs]
# --- Dataset Class ---

class ImageWithReference(Dataset):
    """
    Dataset class for (ref_image,raw_image,toned_image) pairs.
    """
    def __init__(self, image_dir, image_exts=(".jpg", ".png", ".jpeg",".JPG"), have_gt=True, augment=False):
        self.ref_images = self._get_files_ref(image_dir, image_exts)
        self.raw_images = self._get_files_raw(image_dir, image_exts)
        self.have_gt = have_gt
        if have_gt:
            self.toned_images = self._get_files_toned(image_dir, image_exts)
        else:
            self.toned_images = None
        print(f'ImageWithReference ref_images: {len(self.ref_images)}')

    def _get_files_ref(self, root, exts):
        files = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(exts) and 'a' in f:
                    files.append(os.path.join(dirpath, f))
        return sorted(files)
    def _get_files_raw(self, root, exts):
        files = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(exts) and 'b' in f:
                    files.append(os.path.join(dirpath, f))
        return sorted(files)
    def _get_files_toned(self, root, exts):
        files = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(exts) and 'c' in f:
                    files.append(os.path.join(dirpath, f))
        return sorted(files)

    def __len__(self):
        return len(self.ref_images)

    def __getitem__(self, idx):
        img_idx = idx % len(self.ref_images)

        ref_path = self.ref_images[img_idx]
        raw_path = self.raw_images[img_idx]
        if self.have_gt:
            toned_path = self.toned_images[img_idx]
        else:
            toned_path = None

        ref_image = Image.open(ref_path).convert("RGB")
        raw_image = Image.open(raw_path).convert("RGB")
        if self.have_gt:
            toned_image = Image.open(toned_path).convert("RGB")
            ref_image, raw_image, toned_image = resize(ref_image, raw_image, toned_image)
        else:
            toned_image = None
            ref_image, raw_image = resize(ref_image, raw_image)

        return ref_image, raw_image, toned_image


class AcetoneBench(Dataset):
    """
    Dataset class for (ref_image,raw_image,gt_image) pairs.
    """
    def __init__(self, image_dir, image_exts=(".jpg", ".png", ".jpeg"), mode='hard'):
        ref_path = os.path.join(image_dir, 'ref_image')
        raw_path = os.path.join(image_dir, 'raw_image')
        gt_path = os.path.join(image_dir, 'gt_image')
        self.ref_images = self._get_image_files(ref_path, image_exts)
        self.raw_images = self._get_image_files(raw_path, image_exts)
        self.gt_images = self._get_image_files(gt_path, image_exts)
        self.mode = mode
        if self.mode=='simple':
            # print in red
            print(f'\033[91mAcetoneBench in simple mode, the reference is gt_image.\033[0m')
        else:
            # print in green
            print(f'\033[92mAcetoneBench in hard mode, the reference is ref_image.\033[0m') 
        print(f'AcetoneBench images: {len(self.ref_images)}')

    def _get_image_files(self, root, exts):
        files = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(exts):
                    files.append(os.path.join(dirpath, f))
        return sorted(files)

    def __len__(self):
        return len(self.ref_images)

    def __getitem__(self, idx):
        img_idx = idx % len(self.ref_images)

        ref_path = self.ref_images[img_idx]
        raw_path = self.raw_images[img_idx]
        gt_path = self.gt_images[img_idx]

        ref_image = Image.open(ref_path).convert("RGB")
        raw_image = Image.open(raw_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("RGB")
        ref_image, raw_image, gt_image = resize(ref_image, raw_image, gt_image)
        if self.mode=='simple':
            return gt_image, raw_image, gt_image
        else:
            return ref_image, raw_image, gt_image
        
class PSTBench(Dataset):
    """
    Dataset class for (ref_image,raw_image,gt_image) pairs.
    """
    def __init__(self, image_dir, image_exts=(".jpg", ".png", ".jpeg"), mode='hard'):
        ref_path = os.path.join(image_dir, 'paired_style')
        raw_path = os.path.join(image_dir, 'content_709')
        gt_path = os.path.join(image_dir, 'paired_gt')
        self.ref_images = self._get_image_files(ref_path, image_exts)
        self.raw_images = self._get_image_files(raw_path, image_exts)
        self.gt_images = self._get_image_files(gt_path, image_exts)
        self.mode = mode
        if self.mode=='simple':
            # print in red
            print(f'\033[91mPST-Bench in simple mode, the reference is gt_image.\033[0m')
        else:
            # print in green
            print(f'\033[92mPST-Bench in hard mode, the reference is ref_image.\033[0m') 
        print(f'PST-Bench images: {len(self.ref_images)}')

    def _get_image_files(self, root, exts):
        files = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith(exts):
                    files.append(os.path.join(dirpath, f))
        return sorted(files)

    def __len__(self):
        return len(self.ref_images)

    def __getitem__(self, idx):
        img_idx = idx % len(self.ref_images)

        ref_path = self.ref_images[img_idx]
        raw_path = self.raw_images[img_idx]
        gt_path = self.gt_images[img_idx]

        ref_image = Image.open(ref_path).convert("RGB")
        raw_image = Image.open(raw_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("RGB")
        ref_image, raw_image, gt_image = resize(ref_image, raw_image, gt_image)
        if self.mode=='simple':
            return gt_image, raw_image, gt_image
        else:
            return ref_image, raw_image, gt_image
