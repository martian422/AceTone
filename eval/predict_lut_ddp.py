from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from model.modeling_acetone import AceToneModel, AceToneVLM
from model.config_acetone import AceToneConfig
from model.vq import VQVAE3DLUT
from dataset.toning import ImageLUTDataset
from dataset.color_reference import AcetoneBench, PSTBench
from dataset.lut3d import apply_lut, delta_e_between_images, calculate_lpips, calculate_psnr
from color_similarity import ColorSimilarity
import tempfile
import os
import numpy as np
import re
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset.qwen_data import get_path

# default: Load the model on the available device(s)
# DDP setup
local_rank = int(os.environ.get("LOCAL_RANK", -1))
if local_rank != -1:
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    world_size = 1
    rank = 0

def concat_images(*imgs):
    # concat a list of images
    img_concat = Image.new('RGB', (sum(img.width for img in imgs), imgs[0].height))
    x_offset = 0
    for img in imgs:
        img_concat.paste(img, (x_offset, 0))
        x_offset += img.width
    return img_concat.resize((img_concat.width // (len(imgs)//2), img_concat.height // (len(imgs)//2)))

model_dir = get_path("vlm_model_dir", "PATH_TO_ACETONE_VLM_MODEL_DIR")
ckpt_path = get_path("vq_ckpt_path", "PATH_TO_VQ_CKPT_PATH")
acetone_bench_dir = get_path("acetone_bench_dir", "PATH_TO_ACETONE_BENCH_DIR")
pst_bench_dir = get_path("pst_bench_dir", "PATH_TO_PST50_DIR")

# lut_dataset = ImageLUTDataset(image_dir=images_dir, lut_dir=category_lut_dir, augment=True, return_lut_tensor=True)
lut_dataset = AcetoneBench(acetone_bench_dir, mode='hard') # hard is the default mode
# lut_dataset = PSTBench(pst_bench_dir, mode='hard')
dataset_to_process = lut_dataset
sampler = DistributedSampler(dataset_to_process, num_replicas=world_size, rank=rank, shuffle=False) if local_rank != -1 else None


# ________load VQ model_______
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt["model"] if "model" in ckpt else ckpt
vq_model = VQVAE3DLUT(
    codebook_size=ckpt.get("args", {}).get("codebook_size", 256),
    embedding_dim=ckpt.get("args", {}).get("embedding_dim", 64)
)
vq_model.load_state_dict(state_dict)
vq_model.to(device)
vq_model.eval()

# ________load VLM model_______
config = AceToneConfig.from_pretrained(model_dir)
config.model_type = "acetone"
config.mm_vocab_size = 256
model = AceToneVLM.from_pretrained(
    model_dir,
    config=config,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(device)
if local_rank != -1:
    model = DDP(model, device_ids=[local_rank])

# default processor
processor = AutoProcessor.from_pretrained(model_dir)
temp_dir = '/dev/shm' if os.path.exists('/dev/shm') else None
dE = np.array([])
lpips = np.array([])
color_similars = np.array([])
ref_color_similars = np.array([])
psnrs = np.array([])
indices = list(sampler) if sampler else range(len(dataset_to_process))
pbar = tqdm(indices) if rank == 0 else indices
for i in pbar:
    ref_image, ori_image, gt_image = dataset_to_process[i]
    tmp_ori = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=temp_dir)
    ori_image.save(tmp_ori.name)
    tmp_ref = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=temp_dir)
    ref_image.save(tmp_ref.name)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": tmp_ori.name,
                },
                {
                    "type": "image",
                    "image": tmp_ref.name,
                },
                {"type": "text", "text": "The first image is an un-touched raw image, and the second is toned with stylish LUTs. These two images may have the same source or not, and your task is to mimic the toning method. You are a professional color grader. Please generate the 64-bit LUT in \\'Global toning: <SoT>...<EoT>\\'."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    if local_rank != -1:
        generated_ids = model.module.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.01)
    else:
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.01)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    prediction = output_text[0].split('<SoT>')[-1].split('<EoT>')[0]
    # print(prediction)
    prediction_ids_flatten = torch.tensor(np.array(list(map(int, re.findall(r"<MM(\d+)>", prediction)))))
    num_missing = 64 - len(prediction_ids_flatten)
    if num_missing > 0:
        print(f"num_missing={num_missing}")
        last_token = prediction_ids_flatten[0] if len(prediction_ids_flatten) > 0 else torch.tensor(0)
        padding = last_token.repeat(num_missing)
        prediction_ids_flatten = torch.cat([prediction_ids_flatten, padding], dim=0)
    prediction_ids = prediction_ids_flatten[:64].reshape(4,4,4).unsqueeze(0).to(device)
    lut_pred = vq_model.decode_indices(prediction_ids)
    
    lut_pred_image = apply_lut(ori_image, lut_pred.squeeze(0))
    delta_e = delta_e_between_images(gt_image,lut_pred_image)
    lpips_score = calculate_lpips(gt_image,lut_pred_image)
    score = ColorSimilarity(gt_image, lut_pred_image)
    ref_score = ColorSimilarity(ref_image, lut_pred_image)
    psnr = calculate_psnr(gt_image, lut_pred_image)
    dE = np.append(dE, delta_e)
    lpips = np.append(lpips, lpips_score)
    color_similars = np.append(color_similars, score)
    ref_color_similars = np.append(ref_color_similars, ref_score)
    psnrs = np.append(psnrs, psnr)
    if rank == 0:
        print(f"{i}: delta_e={delta_e:.4f}, lpips={lpips_score:.4f}, color_similarity={score:.4f}, ref_color_similarity={ref_score:.4f}, psnr={psnr:.4f}")
    
    # save the 3 images in one.
    if (delta_e < 4 or ref_score > 0.8):
        save_dir_now = f"outputs/good_sample/{i}"
        os.makedirs(save_dir_now, exist_ok=True)
        ref_image.save(f'{save_dir_now}/ref_{i}.png')
        ori_image.save(f'{save_dir_now}/ori_{i}.png')
        lut_pred_image.save(f'{save_dir_now}/lut_pred_{i}.png')
    # print(prediction_ids.reshape(-1))
    # extract code from the output and use vq_model to reconstruct the lut, check its result.
    # ________clean up________
    os.unlink(tmp_ori.name)
    os.unlink(tmp_ref.name)

metrics = {
    "dE": dE,
    "lpips": lpips,
    "color_similars": color_similars,
    "ref_color_similars": ref_color_similars,
    "psnrs": psnrs,
}

if local_rank != -1:
    # Gather metrics from all processes
    gathered_metrics = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_metrics, metrics)

    if rank == 0:
        # Concatenate results from all processes
        all_dE = np.concatenate([m["dE"] for m in gathered_metrics])
        all_lpips = np.concatenate([m["lpips"] for m in gathered_metrics])
        all_color_similars = np.concatenate([m["color_similars"] for m in gathered_metrics])
        all_ref_color_similars = np.concatenate([m["ref_color_similars"] for m in gathered_metrics])
        all_psnrs = np.concatenate([m["psnrs"] for m in gathered_metrics])

        print(f"dE={all_dE.mean():.4f}")
        print(f"lpips={all_lpips.mean():.4f}")
        print(f"color_similars={all_color_similars.mean():.4f}")
        print(f"ref_color_similars={all_ref_color_similars.mean():.4f}")
        print(f"psnrs={all_psnrs.mean():.4f}")
else:
    print(f"dE={dE.mean():.4f}")
    print(f"lpips={lpips.mean():.4f}")
    print(f"color_similars={color_similars.mean():.4f}")
    print(f"ref_color_similars={ref_color_similars.mean():.4f}")
    print(f"psnrs={psnrs.mean():.4f}")

if local_rank != -1:
    dist.destroy_process_group()

# torchrun --nproc_per_node=8 --master_port=23333 eval/predict_lut_ddp.py
