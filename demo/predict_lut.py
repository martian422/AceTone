from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from model.modeling_acetone import AceToneModel, AceToneVLM
from model.config_acetone import AceToneConfig
from model.vq import VQVAE3DLUT
from dataset.toning import ImageLUTDataset
from dataset.color_reference import AcetoneBench, PSTBench
from dataset.lut3d import apply_lut, delta_e_between_images, calculate_lpips, calculate_psnr
from eval.color_similarity import ColorSimilarity
import tempfile
import os
import numpy as np
import re
from PIL import Image
from tqdm import tqdm
from dataset.qwen_data import get_path
# default: Load the model on the available device(s)

def concat_images(*imgs):
    # concat a list of images
    img_concat = Image.new('RGB', (sum(img.width for img in imgs), imgs[0].height))
    x_offset = 0
    for img in imgs:
        img_concat.paste(img, (x_offset, 0))
        x_offset += img.width
    return img_concat.resize((img_concat.width // (len(imgs)//2), img_concat.height // (len(imgs)//2)))

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model_dir = get_path("vlm_model_dir", "PATH_TO_ACETONE_VLM_MODEL_DIR")
category_lut_dir = get_path("luts_npy_c2048_dir", "PATH_TO_LUTS_NPY_C2048_DIR")
images_dir = get_path("adobe5k_extract_dir", "PATH_TO_ADOBE5K_EXTRACT_DIR")
ckpt_path = get_path("vq_ckpt_path", "PATH_TO_VQ_CKPT_PATH")
acetone_bench_dir = 'outputs/workspace/acetone_dataset/acetone-bench-1024/train'
pst_bench_dir = 'outputs/PST50'

# lut_dataset = ImageLUTDataset(image_dir=images_dir, lut_dir=category_lut_dir, augment=True, return_lut_tensor=True)
lut_dataset = AcetoneBench(acetone_bench_dir, mode='hard')
# lut_dataset = PSTBench(pst_bench_dir, mode='hard')


# ________load VQ model_______
device = "cuda" if torch.cuda.is_available() else "cpu"
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
    device_map="auto",
)

# default processor
processor = AutoProcessor.from_pretrained(model_dir)
temp_dir = '/dev/shm' if os.path.exists('/dev/shm') else None
# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
dE = np.array([])
lpips = np.array([])
color_similars = np.array([])
psnrs = np.array([])
for i in tqdm(range(min(1024, len(lut_dataset)))):
    ref_image, ori_image, gt_image = lut_dataset[i]
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
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.1)
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

    # with torch.no_grad():
    #     lut = lut.unsqueeze(0)
    #     _, lut_ids, _, _ = vq_model(lut.to(device))  # (B,C,D,H,W)
    #     lut_ids = lut_ids.cpu().numpy()
    #     lut_ids_flatten = lut_ids.reshape(lut.shape[0], -1)
    #     # print(lut_ids_flatten)
    
    lut_pred_image = apply_lut(ori_image, lut_pred.squeeze(0))
    delta_e = delta_e_between_images(gt_image,lut_pred_image)
    lpips_score = calculate_lpips(gt_image,lut_pred_image)
    score = ColorSimilarity(gt_image, lut_pred_image)
    psnr = calculate_psnr(gt_image, lut_pred_image)
    dE = np.append(dE, delta_e)
    lpips = np.append(lpips, lpips_score)
    color_similars = np.append(color_similars, score)
    psnrs = np.append(psnrs, psnr)
    print(f"{i}: delta_e={delta_e:.4f}, lpips={lpips_score:.4f}, color_similarity={score:.4f}, psnr={psnr:.4f}")
    
    # save the 3 images in one.
    if delta_e < 3:
        img_concat = concat_images(ori_image, lut_pred_image, gt_image)
        img_concat.save('outputs/triples.png')
    # print(prediction_ids.reshape(-1))
    # extract code from the output and use vq_model to reconstruct the lut, check its result.
    # ________clean up________
    os.unlink(tmp_ori.name)
    os.unlink(tmp_ref.name)
print(f"dE={dE.mean():.4f}")
print(f"lpips={lpips.mean():.4f}")
print(f"color_similars={color_similars.mean():.4f}")
print(f"psnrs={psnrs.mean():.4f}")



# for augX, mean=10.69
# for online-acetone, mean=7.96-7.55
# 8.67(0.01) 8.38(0.1) 8.34(0.2)
# after MSCOCO, to 6.9, LPIPS=0.09

# the above are got with augment=False.
breakpoint()
