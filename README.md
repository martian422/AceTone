# AceTone: Bridging Words and Colors for Conditional Image Grading.

English | [中文README](README.zh_CN.md)

This is the official implementation of AceTone (CVPR 2026). Paper: [arXiv:2604.00530](http://arxiv.org/abs/2604.00530).

In this repo, we release:
  1. **LUT tokenizer** for representing LUTs as compact tokens.
  2. **AceTone VLM** modeling for conditional color grading. 
  3. Detailed SFT, RL and Evaluation scripts.

## 🚀 Get Started

First, set up the environment:

```
conda create -n acetone python=3.11
conda activate acetone
```

You shall install pytorch following the [official guide](https://pytorch.org/get-started/locally/).

Then install the remaining packages:

```
pip install -r requirements.txt
```

We use flash attention to accelerate training. You can donwload it from the [official releases](https://github.com/Dao-AILab/flash-attention/releases) and install, or compile it from source:
```
pip install flash_attn==2.7.4.post1 --no-build-isolation
```

If you want to do RL training with GRPO, you shall install trl.
```
pip install trl==0.19.0
```


## 🎨 LUT tokenizer
We propose the AceTone tokenizer so that each 32-bit LUT can be represented with 64 discrete tokens with high fidelity. We have provided the pretrained weights (~4M parameters) in this repo as [model/acetone-vqvae-d64.pt](model/acetone-vqvae-d64.pt).

See [VQ.md](model/VQ.md) for training details.

## 📝 Adding Vocabulary to the VLM
Qwen models have about 300 unused token slots, so for an extended vocabulary size of 256, you can directly modify the tokenizer configuration without editing the LM itself. Please refer to [this script](pipeline/add_vocab.py) for a simple vocabulary extension, and replace the original Qwen tokenizer with the resulted one.

## 🏃 Training AceTone

### Generative Pre-training

We follow the common practice for post-training Qwen models. Here is a guide for customized training:

1. Prepare the dataset configs in [init](dataset/qwen_data/__init__.py).
In this file, the datasets are organized to fetch the image and LUTs from their root folders (these folders can contain subfolders) separately. Training for at least 2 epochs is recommended.
```
ACETONE_PRETRAIN = {
    "image_folder": "YOUR_IMAGE_FOLDER",
    "lut_folder": "YOUR_LUT_FOLDER",
}
```
Theoretically, the more images and LUTs you have (with diverse style and color distributions), the better the resulted model will be. 

If the budget is limited, note that the images' color distribution matters! For certain scenarios like mobile photography, the images may have a significantly different color distribution than the MSCOCO images. Under this circumstance, you may need to collect more photos shot on mobilephones for optimal performance.


2. Run the training script. For the non-selective pretraining setting, we do not organize *image-lut* pairs, but combine them randomly.
```
bash scripts/pretrain_online.sh
```

### Reinforcement learning with GRPO

You may need to organize your RL dataset before GRPO.
```
bash scripts/grpo.sh
```

## 🏁 Evaluation
We provide a script for evaluation. For benchmarking, you can follow the instructions in [predict_lut_ddp.py](eval/predict_lut_ddp.py) to assess the toned results. Just run
```
torchrun --nproc_per_node=8 --master_port=23333 eval/predict_lut_ddp.py
```

## 🛠️ Useful Tools
We provide two scripts for LUT processing in the `useful_tools` directory:
- `convert_luts.py`: This script converts `.cube` files of any size to a uniform 32x32x32 grid, resulting in a NumPy array of shape `(32, 32, 32, 3)`. It saves the output as both `.cube` and `.npy` files.
- `select_luts.py`: This script clusters a large number of LUTs into a smaller, more representative set. This is useful for creating a diverse and high-quality dataset for training.

## 📄 BibTex
If you find this work helpful, please cite:
```
@inproceedings{ma2026acetone,
      title={AceTone: Bridging Words and Colors for Conditional Image Grading},
      author={Tianren Ma and Mingxiang Liao and Xijin Zhang and Qixiang Ye},
      booktitle={CVPR},
      year={2026},
      note={arXiv:2604.00530}
}
```

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
