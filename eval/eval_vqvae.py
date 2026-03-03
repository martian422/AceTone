#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.vq import VQVAE3DLUT   # <-- import your model class here
from dataset.lut3d import LUTDataset, delta_e_between_images_for_luts, generate_standard_image  # <-- replace with your dataset class
from dataset.qwen_data import get_path
import numpy as np
from tqdm import tqdm
from dataset.augment import RandomNoise3D_tensor, RandomIntensityShift_tensor


def psnr(mse, max_val=1.0):
    """Compute PSNR from MSE."""
    return 20 * torch.log10(max_val / torch.sqrt(mse))


@torch.no_grad()
def evaluate(model, dataloader, device="cuda"):
    model.eval()
    total_mse, total_psnr, total_deltaE = np.array([0.0]), np.array([0.0]), np.array([0.0])

    standard_img = generate_standard_image()

    for x in tqdm(dataloader, desc="Evaluating"):
        x = RandomNoise3D_tensor(sigma=0.05)(x)
        x = RandomIntensityShift_tensor(brightness=0.1, contrast=0.1)(x)
        x = x.to(device).float()

        # forward pass
        x_rec, _, _, _ = model(x)

        # mse + psnr
        mse = F.mse_loss(x_rec, x, reduction="none").mean(dim=[1, 2, 3, 4])  # per-sample
        psnr_batch = psnr(mse)
        # deltaE
        # note that current deltaE is downsampled.
        deltaE,_,_ = delta_e_between_images_for_luts(x_rec.permute(0,2,3,4,1).cpu().numpy(), x.permute(0,2,3,4,1).cpu().numpy(), standard_img)
        print(f'deltaE = {deltaE}')
        
        total_deltaE = np.append(total_deltaE, deltaE)
        total_mse = np.append(total_mse, mse.cpu().numpy())
        total_psnr = np.append(total_psnr, psnr_batch.cpu().numpy())

    return total_mse, total_psnr, total_deltaE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_root", type=str, default=get_path("luts_3d_npy_dir", "PATH_TO_LUTS_3D_NPY_DIR"), help="Dataset root folder")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension, remember to set this to the same value as in training")
    parser.add_argument("--codebook_size", type=int, default=256, help="Codebook size, remember to set this to the same value as in training")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # build dataset + dataloader
    dataset = LUTDataset(args.data_root)  # <-- customize
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # build model
    model = VQVAE3DLUT(codebook_size=args.codebook_size, embedding_dim=args.embedding_dim)
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)

    model.to(device)
    
    # run evaluation
    total_mse, total_psnr, total_deltaE = evaluate(model, dataloader, device=device)

    print(f"Reconstruction MSE:  {total_mse.mean():.6f}")
    print(f"Reconstruction PSNR: {total_psnr.mean():.2f} dB")
    print(f"Reconstruction deltaE: {total_deltaE.mean():.2f}")
    print(f"Reconstruction worse deltaE (max): {total_deltaE.max():.2f}")


if __name__ == "__main__":
    main()
