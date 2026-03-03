### Training:
If you want to train your own tokenizer, please organize your LUTs as .npy files under a folder (shape (32,32,32,3) or (3,32,32,32) in [0,1]).

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_vq.py --data_dir /path/to/luts --augment
```
Tweakables:
--codebook_size (e.g., 256–2048, default 256), --embedding_dim (32–128, default 64), --beta (commitment, default 0.25), and --vq_weight(default 3e-4).

Latent grid is 4×4×4 by design (from 32→16→8→4). If you want fewer tokens, add another downsample stage; If you want more fidelity, remove one.

The default setting can yield a model with average delta E < 2, good to start with.

### Evaluation:

Eval (with random noise):
```
python eval/eval_hard.py --ckpt /path/to/ckpt --data_root /path/to/luts
```
Visualize:
```
python demo/sample_and_apply.py
```