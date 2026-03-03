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

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dataset.lut3d import LUTDataset  # your dataset
from model.vq import VQVAE3DLUT
from dataset.lut3d import apply_lut, delta_e_between_images  # <-- use your functions
from tqdm import tqdm
from dataset.qwen_data import get_path

# -----------------------------
# Parameters
# -----------------------------
image_path = "eval/samples/sailor.jpg"
ckpt_path = get_path("vq_ckpt_path", "PATH_TO_VQ_CKPT_PATH")

N = 6
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load image & model
# -----------------------------
I = Image.open(image_path).convert("RGB")

ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt["model"] if "model" in ckpt else ckpt

model = VQVAE3DLUT(
    codebook_size=ckpt.get("args", {}).get("codebook_size", 256),
    embedding_dim=ckpt.get("args", {}).get("embedding_dim", 64)
)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# -----------------------------
# Sample LUTs
# -----------------------------
lut_dataset = LUTDataset(augment=False)
np.random.seed(42)
idxs = np.random.choice(len(lut_dataset), size=N, replace=False)
luts = [lut_dataset[i] for i in idxs]  # (3, 32, 32, 32)

# -----------------------------
# Apply LUTs + Reconstruct
# -----------------------------
results = []
i=0
for lut in tqdm(luts, desc="Applying LUTs"):
    lut_t = lut.unsqueeze(0).to(device)

    with torch.no_grad():
        lut_rec, _, _, _ = model(lut_t)  # (B,C,D,H,W)
        lut_rec = lut_rec.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        lut_rec = np.clip(lut_rec, 0, 1)

    # apply LUTs
    img_orig = apply_lut(I, lut)
    img_rec = apply_lut(I, lut_rec)

    img_rec.save(f'outputs/images/styles/lut_rec_{i}.png')

    i+=1
    # compute ΔE
    delta_e = delta_e_between_images(img_orig, img_rec)

    results.append((img_orig, img_rec, delta_e))

# -----------------------------
# Plot Results
# -----------------------------
# -----------------------------
# Plot Results (4 images per row: 2 before-after pairs)
# -----------------------------
pairs_per_row = 2
images_per_row = pairs_per_row * 2  # each pair has 2 images
rows = int(np.ceil(N / pairs_per_row))

fig, axes = plt.subplots(rows, images_per_row, figsize=(10, 2 * rows))

if rows == 1:
    axes = np.expand_dims(axes, 0)  # ensure 2D array

for i, (img_o, img_r, dE) in enumerate(results):
    row = i // pairs_per_row
    col = (i % pairs_per_row) * 2

    # Original image
    axes[row, col].imshow(img_o)
    axes[row, col].set_title(f"LUT {i+1} (Original)")
    axes[row, col].axis("off")

    # Reconstructed image
    axes[row, col + 1].imshow(img_r)
    axes[row, col + 1].set_title(f"Reconstructed, ΔE={dE:.2f}")
    axes[row, col + 1].axis("off")

# Hide unused subplots (if N is not multiple of pairs_per_row)
for j in range(i + 1, rows * pairs_per_row):
    row = j // pairs_per_row
    col = (j % pairs_per_row) * 2
    axes[row, col].axis("off")
    axes[row, col + 1].axis("off")

plt.tight_layout()
# save as pdf
plt.savefig("outputs/tokenizer.pdf", dpi=150)
plt.show()
