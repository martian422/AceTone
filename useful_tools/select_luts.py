import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap.umap_ as umap
from scipy.ndimage import zoom
from tqdm import tqdm
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from skimage import color

test_image = Image.open("eval/samples/sailor.jpg").convert("RGB")
TEST_IMG = test_image.resize((200,200))

def apply_lut(image, lut):
    """
    Apply a 3D LUT to an image.

    Args:
        image: PIL Image object
        lut: 3D numpy array of shape (size, size, size, 3)

    Returns:
        PIL Image object with LUT applied
    """
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

def not_bw(img, gray_tolerance=15, gray_fraction=0.90):
    """
    Check if a PIL image is approximately black and white.

    Parameters:
        img (PIL.Image): Input image.
        gray_tolerance (int): Max RGB deviation per pixel to still count as gray.
        gray_fraction (float): Minimum fraction of grayish pixels to consider B&W.

    Returns:
        bool: True if the image is approximately black and white, False otherwise.
    """
    
    # Convert to numpy array
    arr = np.array(img, dtype=np.int16)  # int16 to prevent overflow
    
    # Compute per-pixel color deviation
    diff = arr.max(axis=2) - arr.min(axis=2)
    
    # Fraction of pixels within gray tolerance
    grayish_ratio = np.mean(diff <= gray_tolerance)
    
    return grayish_ratio < gray_fraction

def is_not_minor(ori_image, lut_image):
    de = delta_e_between_images(ori_image, lut_image)
    return de > 3

def load_and_downsample(input_dir, downsample_size=16):
    lut_arrays = []
    file_names = []
    npy_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    bad = 0
    for path in tqdm(npy_files, desc="Loading LUT files"):
        try:
            lut = np.load(path)
            check = apply_lut(TEST_IMG,lut)
            if lut.shape == (32, 32, 32, 3) and not_bw(check) and is_not_minor(TEST_IMG,check):
                factors = (downsample_size/32, downsample_size/32, downsample_size/32, 1)
                lut_small = zoom(lut, factors, order=1)  # Downsample with linear interpolation
                lut_arrays.append(lut_small.reshape(-1))
                file_names.append(os.path.relpath(path, input_dir))
            else:
                bad += 1
                print(f"[Skipping] {path} shape {lut.shape} is not (32,32,32,3), or it's B&W, or the difference is too small.")
        except Exception as e:
            print(f"[Error] {path}: {e}")
    print(f"[INFO] Skipped a total of {bad} files.")
    return np.array(lut_arrays), file_names

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from tqdm import tqdm
import shutil

def cluster_and_extract(
    lut_arrays, file_names, input_dir, output_dir,
    n_clusters=256, pca_dim=128
):
    """
    Cluster LUTs into n_clusters, extract representatives,
    and save them into output_dir.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: PCA dimensionality reduction
    print(f"[INFO] Reducing to {pca_dim} dimensions with PCA...")
    pca = PCA(n_components=pca_dim, random_state=42)
    reduced = pca.fit_transform(lut_arrays)

    # Step 2: KMeans clustering
    print(f"[INFO] Clustering into {n_clusters} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42, batch_size=512, n_init="auto"
    )
    labels = kmeans.fit_predict(reduced)

    # Step 3: Find representative LUTs
    print("[INFO] Selecting representatives...")
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, reduced)
    representatives = [file_names[idx] for idx in closest]

    # Step 4: Save representative LUTs with labels

    for i, rep in tqdm(enumerate(representatives), desc="Saving representatives"):
        src_path = os.path.join(input_dir, rep)
        name = rep.split('/')[-1].split('.')[0]
        dst_path = os.path.join(output_dir, f'class_{i}_{name}.npy')

        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"[ERROR] Could not copy {src_path} → {dst_path}: {e}")

    print(f"[DONE] Saved {len(representatives)} representative LUTs to {output_dir}")

    return labels, representatives


input_dir = "datasets/LUTS-final/cube-3d-npy"

output_dir = "datasets/LUTS-final/cube-npy-c8192"

lut_arrays, file_names = load_and_downsample(input_dir, downsample_size=16)
labels, representatives = cluster_and_extract(
    lut_arrays, file_names, input_dir, output_dir,
    n_clusters=8192
)

print("Samples:")
for r in representatives[:10]:
    print(r)