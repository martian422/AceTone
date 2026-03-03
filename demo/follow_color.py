from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from model.modeling_acetone import AceToneModel, AceToneVLM
from model.config_acetone import AceToneConfig
from model.vq import VQVAE3DLUT
from dataset.color_reference import ImageWithReference
from dataset.lut3d import apply_lut, apply_lut_rev, calculate_lpips, delta_e_between_images
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
images_dir = get_path("toned_images_dir", "PATH_TO_TONED_IMAGES_DIR")
ckpt_path = get_path("vq_ckpt_path", "PATH_TO_VQ_CKPT_PATH")

ref_dataset = ImageWithReference(image_dir=images_dir, have_gt=True)

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
lpips = np.array([])
delta_e = np.array([])
for i in tqdm(range(len(ref_dataset))):
    ref_image, raw_image, toned_image = ref_dataset[i]
    tmp_ref = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=temp_dir)
    ref_image.save(tmp_ref.name)
    tmp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=temp_dir)
    raw_image.save(tmp_raw.name)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": tmp_raw.name,
                },
                {
                    "type": "image",
                    "image": tmp_ref.name,
                },
                {"type": "text", "text": "The first image is un-touched, and the second image is toned with stylish LUTs. These two images may have the same source or not, and your task is to mimic the toning method. You are a professional color grader. Please generate the 64-bit LUT in \\'Global toning: <SoT>...<EoT>\\'."},
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
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.05)
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
        # lut = lut.unsqueeze(0)
        # _, lut_ids, _, _ = vq_model(lut.to(device))  # (B,C,D,H,W)
        # lut_ids = lut_ids.cpu().numpy()
        # lut_ids_flatten = lut_ids.reshape(lut.shape[0], -1)
        # print(lut_ids_flatten)
    
    lut_pred_image = apply_lut(raw_image, lut_pred.squeeze(0))
    if toned_image is not None:
        lpips = np.append(lpips, calculate_lpips(toned_image, lut_pred_image))
        delta_e = np.append(delta_e, delta_e_between_images(toned_image, lut_pred_image))
        print(f"lpips={lpips[-1]:.4f}, delta_e={delta_e[-1]:.4f}")
        result = concat_images(ref_image, raw_image, toned_image, lut_pred_image)
    else:
        result = concat_images(ref_image, raw_image, lut_pred_image)
    result.save(f'outputs/base_follow_test/base1_result_{i}.png')
    # print(prediction_ids.reshape(-1))
    # breakpoint()
    # extract code from the output and use vq_model to reconstruct the lut, check its result.
    # ________clean up________
    os.unlink(tmp_ref.name)
    os.unlink(tmp_raw.name)

print(f"Final Average lpips={lpips.mean():.4f}")
print(f"Final Average delta_e={delta_e.mean():.4f}")
