"""
Simple VQ‑VAE for 3D LUTs (PyTorch)
------------------------------------

• Assumes each LUT is a numpy .npy file shaped (32, 32, 32, 3) or (3, 32, 32, 32).
• Trains a 3D‑conv VQ‑VAE that compresses the LUT volume into a grid of discrete codes.
• The latent grid is 4×4×4 (64 tokens). Each token is an index in a codebook (default K=512).
• You can later model the sequence of 64 indices with an autoregressive model.

Usage
-----
python vqvae_3dlut.py \
  --data_dir /path/to/luts_npy \
  --epochs 50 --batch_size 8 --lr 3e-4 \
  --codebook_size 512 --embedding_dim 64 \
  --save_dir ./runs/vqvae_lut

After training, see saved samples and a checkpoint in save_dir.
You can also run encode/decode to get indices or reconstruct LUTs.

Note: This is a compact, readable reference—tune architecture/hparams for best PSNR/ΔE.
"""

import argparse
import os
import math
import time
from pathlib import Path
from typing import Tuple
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.lut3d import LUTDataset
from dataset.qwen_data import get_path
from model.vq import VQVAE3DLUT
# -----------------------------
# Training utils
# -----------------------------

def psnr(x, y, eps=1e-8):
    mse = F.mse_loss(x, y, reduction='none').mean(dim=(1,2,3,4))
    return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))


def train(args):
    # print some training info in green
    print(f"\033[92mTraining VQ-VAE (EMA version) for 3D LUTs with codebook size {args.codebook_size} and embedding dim {args.embedding_dim}\033[0m")
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    if args.augment:
        print("Using data augmentation")
        save_dir = os.path.join(args.save_dir,f"VQ-AUG-c{args.codebook_size}-d{args.embedding_dim}")
    else:
        print("Not using data augmentation")
        save_dir = os.path.join(args.save_dir,f"VQ-c{args.codebook_size}-d{args.embedding_dim}")
    os.makedirs(save_dir, exist_ok=True)

    ds = LUTDataset(args.data_dir, augment=args.augment)
    n_train = int(len(ds) * (1 - args.val_split))
    n_val = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    print(f"Using {device} for training for {args.epochs} epochs in total")
    print(f"Training on {n_train} samples, validating on {n_val} samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = VQVAE3DLUT(
        codebook_size=args.codebook_size,
        embedding_dim=args.embedding_dim,
        beta=args.beta,
        hidden=args.hidden,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)

    global_step = 0
    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = {"loss":0.0, "recon":0.0, "vq":0.0, "psnr":0.0, "perplexity":0.0}
        for x in train_loader:
            x = x.to(device)
            x_rec, _, loss_vq, perplexity = model(x)
            loss_recon = F.mse_loss(x_rec, x)
            loss = loss_recon + args.vq_weight * loss_vq

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                p = psnr(x_rec, x).mean()
            running["loss"] += loss.item()
            running["recon"] += loss_recon.item()
            running["vq"] += loss_vq.item()
            running["psnr"] += p.item()
            running["perplexity"] += perplexity.item()
            global_step += 1
        n_batches = len(train_loader)
        log = {k: v/max(1,n_batches) for k,v in running.items()}

        # Validation
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                x_rec, _, _, _ = model(x)
                val_psnr += psnr(x_rec, x).mean().item()
        val_psnr /= max(1,len(val_loader))

        # save the log locally
        with open(os.path.join(save_dir, "log.txt"), "a") as f:
            f.write(f"Epoch {epoch},loss {log['loss']},recon {log['recon']},vq_loss {log['vq']},PSNR {log['psnr']},PPL {log['perplexity']},Val_PSNR {val_psnr}.\n")

        print(f"Epoch {epoch:03d} | {time.time()-t0:.1f}s | loss {log['loss']:.5f} | recon {log['recon']:.5f} | vq {log['vq']:.5f} | psnr {log['psnr']:.2f} | perplexity {log['perplexity']:.2f} | val_psnr {val_psnr:.2f}")

        # Save checkpoint
        ckpt = {
            'model': model.state_dict(),
            'args': vars(args),
            'epoch': epoch,
            'val_psnr': val_psnr,
        }
        model_config = {
            'codebook_size': args.codebook_size,
            'embedding_dim': args.embedding_dim,
            'beta': args.beta,
            'hidden': args.hidden,
        }
        # use json to save model config
        with open(os.path.join(save_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f)

        if epoch % 50 == 0:
            torch.save(ckpt, os.path.join(save_dir, f"ckpt_epoch{epoch:03d}.pt"))

        if val_psnr > best_val:
            best_val = val_psnr
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))

    print("Training complete. Best val PSNR:", best_val)

# -----------------------------
# Encode/Decode helpers
# -----------------------------

def load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    a = ckpt['args']
    model = VQVAE3DLUT(
        codebook_size=a['codebook_size'],
        embedding_dim=a['embedding_dim'],
        beta=a['beta'],
        hidden=a['hidden'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def export_indices(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = load_model(args.checkpoint, device)
    ds = LUTDataset(args.data_dir, auto_rescale=args.auto_rescale)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    all_indices = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            idx = model.encode_indices(x)  # (B,4,4,4)
            all_indices.append(idx.cpu().numpy())
    indices = np.concatenate(all_indices, axis=0)
    out = os.path.join(args.save_dir, 'codes.npy')
    os.makedirs(args.save_dir, exist_ok=True)
    np.save(out, indices)
    print(f"Saved codes to {out} with shape {indices.shape} and dtype {indices.dtype}")


def reconstruct_from_indices(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = load_model(args.checkpoint, device)
    codes = np.load(args.codes)
    if codes.ndim == 3:
        codes = codes[None, ...]
    indices = torch.from_numpy(codes).long().to(device)
    with torch.no_grad():
        x_rec = model.decode_indices(indices).cpu().numpy()
    out = os.path.join(args.save_dir, 'recon_from_codes.npy')
    np.save(out, x_rec)
    print(f"Saved recon LUT(s) to {out} with shape {x_rec.shape}")

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default=get_path("luts_3d_npy_dir", "PATH_TO_LUTS_3D_NPY_DIR"))
    p.add_argument('--save_dir', type=str, default='./outputs/ckpt')
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--embedding_dim', type=int, default=64)
    p.add_argument('--codebook_size', type=int, default=256)
    p.add_argument('--beta', type=float, default=0.25, help='commitment weight for VQ')
    p.add_argument('--vq_weight', type=float, default=1e-2, help='scale VQ loss vs recon')
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--augment', action='store_true', help='data augmentation')
    p.add_argument('--cpu', action='store_true')
    # Export/Decode modes
    p.add_argument('--checkpoint', type=str, default='')
    p.add_argument('--export_codes', action='store_true')
    p.add_argument('--codes', type=str, default='')
    p.add_argument('--reconstruct_from_codes', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    if args.export_codes:
        if not args.checkpoint:
            raise ValueError('--export_codes requires --checkpoint')
        export_indices(args)
        return
    if args.reconstruct_from_codes:
        if not args.checkpoint or not args.codes:
            raise ValueError('--reconstruct_from_codes requires --checkpoint and --codes')
        reconstruct_from_indices(args)
        return
    train(args)


if __name__ == '__main__':
    main()
