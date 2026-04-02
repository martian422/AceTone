# AceTone：Bridging Words and Colors for Conditional Image Grading.

这是 AceTone（CVPR 2026）的官方实现。论文：[arXiv:2604.00530](http://arxiv.org/abs/2604.00530)。

本仓库发布内容包括：
  1. 用于将 LUT 表示为紧凑 token 的 **LUT tokenizer**。
  2. 用于条件调色的 **AceTone VLM** 建模。
  3. 完整的 SFT、RL 与评测脚本。

## 🚀 快速开始

首先，创建并激活环境：

```
conda create -n acetone python=3.11
conda activate acetone
```

请按照 [PyTorch 官方指南](https://pytorch.org/get-started/locally/) 安装 pytorch。

然后安装其余依赖：

```
pip install -r requirements.txt
```

我们使用 flash attention 加速训练。你可以从 [官方 releases](https://github.com/Dao-AILab/flash-attention/releases) 下载并安装，或从源码编译安装：
```
pip install flash_attn==2.7.4.post1 --no-build-isolation
```

如果你希望使用 GRPO 进行 RL 训练，需要安装 trl：
```
pip install trl==0.19.0
```

## 🎨 LUT tokenizer

我们提出 AceTone tokenizer，使得每个 32-bit LUT 可以用 64 个离散 token 高保真地表示。预训练权重（约 4M 参数）已包含在本仓库中，在 [model/acetone-vqvae-d64.pt](model/acetone-vqvae-d64.pt)。

训练细节请参见 [VQ.md](model/VQ.md)。

## 📝 向 VLM 添加词表

Qwen 模型大约有 300 个未使用的 token 槽位，因此当扩展词表大小为 256 时，你可以直接修改 tokenizer 配置，而无需编辑 LM 本体。可参考 [该脚本](pipeline/add_vocab.py) 完成简单的词表扩展，并用生成的新 tokenizer 替换原始 Qwen tokenizer。

## 🏃 训练 AceTone

### 生成式预训练

我们遵循 Qwen 模型后训练（post-training）的通用实践。以下是自定义训练指南：

1. 在 [init](dataset/qwen_data/__init__.py) 中准备数据集配置。
在该文件里，数据集被组织为分别从各自的根目录（根目录内可以包含子目录）读取图像与 LUT。建议训练至少 2 个 epoch。
```
ACETONE_PRETRAIN = {
    "image_folder": "YOUR_IMAGE_FOLDER",
    "lut_folder": "YOUR_LUT_FOLDER",
}
```
理论上，图像与 LUT 越多（风格与颜色分布越多样），最终模型效果越好。

如果预算有限，请注意图像的颜色分布很关键！例如在移动摄影等场景下，图像的颜色分布可能与 MSCOCO 图像显著不同。在这种情况下，为获得最佳性能，你可能需要收集更多手机拍摄照片。

2. 运行训练脚本。在非选择性（non-selective）的预训练设置下，我们不组织 *image-lut* 配对，而是随机组合它们。
```
bash scripts/pretrain_online.sh
```

### 使用 GRPO 的强化学习

在使用 GRPO 前，你可能需要先组织你的 RL 数据集。
```
bash scripts/grpo.sh
```

## 🏁 评测

我们提供了评测脚本。做基准测试时，你可以按照 [predict_lut_ddp.py](eval/predict_lut_ddp.py) 中的说明评估调色结果。直接运行：
```
torchrun --nproc_per_node=8 --master_port=23333 eval/predict_lut_ddp.py
```

## 🛠️ 实用工具
我们在 `useful_tools` 目录中提供了两个用于 LUT 处理的脚本：
- `convert_luts.py`: 此脚本将任何大小的 `.cube` 文件统一转换为 32x32x32 的网格，生成一个形状为 `(32, 32, 32, 3)` 的 NumPy 数组。它将输出同时保存为 `.cube` 和 `.npy` 文件。
- `select_luts.py`: 此脚本将大量的 LUT 文件聚类成一个更小、更具代表性的集合。这对于为训练创建多样化、高质量的数据集非常有用。

## 📄 BibTex

如果你觉得本工作有帮助，请引用：
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

本项目遵循 [Apache-2.0 License](LICENSE)。

