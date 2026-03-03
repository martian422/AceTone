# Code and weights
# Modified from https://github.com/ZHKKKe/NeuralPreset/blob/main/src/metric/style_similiary/calc_style_similiary.py

import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import onnxruntime
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------
# Core function
# ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_onnx_session():
    """Load ONNX model only once and reuse it."""
    onnx_path = 'outputs/pretrained/StyleSimiliaryDiscriminator.onnx'
    return onnxruntime.InferenceSession(onnx_path)

def ColorSimilarity(img1: Image.Image, img2: Image.Image) -> float:
    """
    Calculate the color style similarity score between two images.

    Args:
        img1 (PIL.Image): First image (e.g., result image).
        img2 (PIL.Image): Second image (e.g., style reference image).

    Returns:
        float: similarity score.
    """
    # Preprocess images
    img1_arr = np.asarray(img1.convert("RGB").resize((512, 512))).astype(np.float32)
    img2_arr = np.asarray(img2.convert("RGB").resize((512, 512))).astype(np.float32)

    # Run inference
    sess = get_onnx_session()
    score = sess.run(['score'], {'ref': img2_arr, 'img': img1_arr})[0]

    return float(score.squeeze())

# ----------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------

if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-folder', type=str, default='outputs/images/results')
    parser.add_argument('--style-folder', type=str, default='outputs/images/styles')
    args = parser.parse_args()

    # check cmd arguments
    if not os.path.exists(args.result_folder):
        print(f'Cannot find the result folder: {args.result_folder}')
        exit()
    if not os.path.exists(args.style_folder):
        print(f'Cannot find the style folder: {args.style_folder}')
        exit()

    input_files = os.listdir(args.result_folder)
    style_files = os.listdir(args.style_folder)

    print('--------------------------------------------------------------------------------')
    print(f'result folder: {args.result_folder}')
    print(f'style folder: {args.style_folder}')
    print('--------------------------------------------------------------------------------')

    # calculate style similarity score
    style_similarity_scores = []
    imgs=[]
    pbar = tqdm(input_files, total=len(input_files), unit='file')
    for fname in pbar:
        result_path = os.path.join(args.result_folder, fname)
        # style_path = os.path.join(args.style_folder, fname)
        style_path = 'outputs/images/styles/lut_rec_1.png'

        result_img = Image.open(result_path)
        style_img = Image.open(style_path)
        imgs.append(result_img)

        # use the function
        score = ColorSimilarity(result_img, style_img)
        style_similarity_scores.append(score)

    N = len(imgs)
    fig, axes = plt.subplots(N, 2, figsize=(7, 3 * N))
    if N == 1:
        axes = [axes]

    for i, img in enumerate(imgs):
        axes[i][0].imshow(style_img)
        axes[i][0].set_title("Style image")
        axes[i][0].axis("off")

        axes[i][1].imshow(img)
        axes[i][1].set_title(f"Query image {i+1} (Score={style_similarity_scores[i]:.4f})")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/images/color_similarity.png", dpi=96)
