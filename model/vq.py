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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    """
    EMA version of VQ (from VQ-VAE v2).
    Maintains codebook via exponential moving average updates.
    """
    def __init__(self, codebook_size, embedding_dim, beta=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.K = codebook_size
        self.D = embedding_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps

        embed = torch.randn(self.K, self.D)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(self.K))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z_e):
        """
        z_e: (B, D, Z, Y, X)
        Returns: z_q_st, indices, vq_loss
        """
        B, D, Z, Y, X = z_e.shape
        ze_flat = z_e.permute(0,2,3,4,1).reshape(-1, D)  # (N, D)

        # Compute distances
        dist = (
            ze_flat.pow(2).sum(1, keepdim=True)
            - 2 * ze_flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )  # (N, K)

        indices = torch.argmin(dist, dim=1)  # (N,)
        z_q = F.embedding(indices, self.embedding)  # (N, D)
        z_q = z_q.view(B, Z, Y, X, D).permute(0,4,1,2,3)  # (B,D,Z,Y,X)
        indices = indices.view(B, Z, Y, X)
        # indices: (B, Z, Y, X)
        flat_indices = indices.view(-1)

        # Histogram of code usage in the current batch
        encodings_one_hot = F.one_hot(flat_indices, self.K).float()
        avg_probs = encodings_one_hot.mean(0)   # (K,)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if self.training:
            with torch.no_grad():
                # EMA updates
                one_hot = F.one_hot(indices.view(-1), self.K).type_as(z_e)  # (N,K)
                cluster_size = one_hot.sum(0)

                # Update cluster size and embed_avg
                self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                embed_sum = ze_flat.t() @ one_hot  # (D,K)
                self.embed_avg.mul_(self.decay).add_(embed_sum.t(), alpha=1 - self.decay)

                # Normalized embedding update
                n = self.cluster_size.sum()
                cluster_size = (self.cluster_size + self.eps) / (n + self.K * self.eps) * n
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
                self.embedding.copy_(embed_normalized)

        # Straight-through trick
        z_q_st = z_e + (z_q - z_e).detach()

        # VQ loss (only commitment, codebook updated via EMA)
        loss_vq = self.beta * F.mse_loss(z_e, z_q.detach())

        # Codebook usage (perplexity)
        flat_indices = indices.view(-1)
        encodings_one_hot = F.one_hot(flat_indices, self.K).float()
        avg_probs = encodings_one_hot.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, indices, loss_vq, perplexity

    @torch.no_grad()
    def compute_codes(self, z_e):
        """Get hard indices (no ST)."""
        B, D, Z, Y, X = z_e.shape
        ze_flat = z_e.permute(0,2,3,4,1).reshape(-1, D)
        dist = (
            ze_flat.pow(2).sum(1, keepdim=True)
            - 2 * ze_flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )
        indices = torch.argmin(dist, dim=1)
        z_q = F.embedding(indices, self.embedding)
        z_q = z_q.view(B, Z, Y, X, D).permute(0,4,1,2,3)
        indices = indices.view(B, Z, Y, X)
        return indices, z_q

# -----------------------------
# Encoder / Decoder (3D)
# -----------------------------

class Encoder3D(nn.Module):
    def __init__(self, in_channels=3, hidden=64, embedding_dim=64):
        super().__init__()
        # Downsample 32->16->8->4 with stride 2
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(hidden, hidden*2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden*2, hidden*2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(hidden*2, hidden*4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden*4, embedding_dim, 1),  # project to D
        )

    def forward(self, x):
        # x: (B,3,32,32,32) -> (B,D,4,4,4)
        return self.net(x)

class Decoder3D(nn.Module):
    def __init__(self, out_channels=3, hidden=64, embedding_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(embedding_dim, hidden*4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(hidden*4, hidden*2, 4, stride=2, padding=1),  # 4->8
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden*2, hidden*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(hidden*2, hidden, 4, stride=2, padding=1),   # 8->16
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(hidden, out_channels, 4, stride=2, padding=1),  # 16->32
            nn.Sigmoid(),  # outputs in [0,1]
        )

    def forward(self, z_q):
        return self.net(z_q)

# -----------------------------
# VQ‑VAE Model
# -----------------------------

class VQVAE3DLUT(nn.Module):
    def __init__(self, codebook_size=256, embedding_dim=64, beta=0.25, hidden=64):
        super().__init__()
        self.encoder = Encoder3D(in_channels=3, hidden=hidden, embedding_dim=embedding_dim)
        self.vq = VectorQuantizerEMA(codebook_size, embedding_dim, beta)
        self.decoder = Decoder3D(out_channels=3, hidden=hidden, embedding_dim=embedding_dim)
        self.device = self.encoder.net[0].weight.device

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, indices, loss_vq, perplexity = self.vq(z_e)
        x_rec = self.decoder(z_q)
        return x_rec, indices, loss_vq, perplexity

    @torch.no_grad()
    def encode_indices(self, x):
        z_e = self.encoder(x)
        indices, _ = self.vq.compute_codes(z_e)
        return indices  # (B,4,4,4)

    @torch.no_grad()
    def decode_indices(self, indices):
        # Map indices -> embeddings -> decode
        e = self.vq.embedding  # (K,D)
        B, Z, Y, X = indices.shape
        D = e.shape[1]
        z_q = F.embedding(indices.view(-1), e).view(B, Z, Y, X, D).permute(0,4,1,2,3)
        x_rec = self.decoder(z_q)
        return x_rec