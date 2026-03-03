# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataset.lut3d import apply_lut
from dataset.toning import ImageLUTDataset_for_RL
from PIL import Image
import os
from model.vq import VQVAE3DLUT
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import copy
from dataset.qwen_data import get_path

def get_save_folder(base_dir, index):
    """Return subfolder path based on image index, create if not exists."""
    max_per_folder = 1000
    folder_idx = index // max_per_folder
    folder_name = f"{folder_idx:05d}"   # e.g. 00000, 00001
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_name


category_lut_dir = get_path("luts_npy_c8192_dir", "PATH_TO_LUTS_NPY_C8192_DIR")
images_dir = get_path("adobe5k_extract_dir", "PATH_TO_ADOBE5K_EXTRACT_DIR")
image_save_path = 'outputs/workspace/acetone_dataset/images/rl-8k'
jsonl_save_path = 'outputs/workspace/acetone_dataset/annotations/rl-8k.jsonl'

os.makedirs(image_save_path, exist_ok=True)
template = {
      "image": ["ref_img", "ori_img", "gt_img"]
    }

# lut_dataset = ImageLUTDataset(image_dir=images_dir, lut_dir=category_lut_dir, return_lut_tensor=True, augment=True)
lut_dataset = ImageLUTDataset_for_RL(image_dir=images_dir, lut_dir=category_lut_dir, augment=True, shuffle=True)
# the dataset returns (image(PIL), toned_image(PIL), img_path, lut_path)

lut_dataset.shuffle()
img_counter = 0
with open(jsonl_save_path, "a", encoding="utf-8") as f_jsonl:
    for i in tqdm(range(65536), desc="Rendering"):
        (ref_image, ori_image, gt_image, result_id) = lut_dataset[i]
        save_folder = get_save_folder(image_save_path, img_counter)
        ref_path = os.path.join(save_folder, f'{result_id}_ref.jpg')
        before_path = os.path.join(save_folder, f'{result_id}_ori.jpg')
        after_path = os.path.join(save_folder, f'{result_id}_gt.jpg')

        # save immediately
        ref_image.save(os.path.join(image_save_path, ref_path))
        ori_image.save(os.path.join(image_save_path, before_path))
        gt_image.save(os.path.join(image_save_path, after_path))

        cur_conv = copy.deepcopy(template)
        cur_conv["image"] = [ref_path, before_path, after_path]
        f_jsonl.write(json.dumps(cur_conv, ensure_ascii=False) + "\n")

        img_counter += 1
