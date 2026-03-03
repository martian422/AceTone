# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from typing import List, Dict, Any

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration
import pathlib

from grpo.trainer import AcetoneGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

import numpy as np
# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple

from model.vq import VQVAE3DLUT
from dataset.lut3d import apply_lut, delta_e_between_images
from eval.color_similarity import ColorSimilarity
from dataset.qwen_data import get_path

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    score_reward_threshold: Optional[float] = field(
        default=0.35,
        metadata={"help": "Threshold for score reward"},
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "JSON file path for the dataset"},
    )

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

COMPARE_QUESTION_PROMPT =     (
    "Decide which enhanced image is superior "
    "or if they are comparable. Evaluate based on: "
    "1) fidelity and consistency with the reference image; "
    "2) overall perceptual quality. "
    "Return **exactly one** of: Image A, Image B, or Similar."
)

SYS = (
    "You are a helpful assistant. "
)
MY_QUESTION = (
    "The first image is an un-touched raw image, "
    "and the second is toned with stylish LUTs. "
    "These two images may have the same source or not, and your task is to mimic the toning method. "
    "You are a professional color grader. Please generate a 64-bit LUT in \\'Global toning: <SoT>...<EoT>\\'."
)


class LazyComparisonDataset(Dataset):
    def __init__(self, script_args: GRPOScriptArguments):
        super().__init__()
        self.script_args = script_args

        # Only supports datasets for a single image comparison task
        json_path = getattr(script_args, "dataset", None)
        if not json_path:
            raise ValueError("Please provide the dataset file: --dataset <path_to_json>")
        self.samples = self._load_samples_auto(json_path)
        self.total_len = len(self.samples)

    def _load_samples_auto(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Automatically loads dataset samples. Supports .jsonl/.json index files or an image directory.

        ASSUMPTION: If data_path is a directory, it is assumed to be the same as self.script_args.image_root.
        """
        print(f"Checking dataset from {data_path}")
        
        valid_samples = []
        
        # ------------------ New: Handle Directory (Simplified) ------------------
        if os.path.isdir(data_path):
            print(f"Detected folder path. Loading samples directly from the root directory.")
            
            # Define the file suffixes and their order in the list
            # SUFFIXES = ['ref', 'ori', 'gt'] 
            SUFFIXES = ['a', 'b', 'c'] 
            files = os.listdir(data_path)
            grouped_files: Dict[str, List[str]] = {}
            
            # In this simplified case, root is data_path
            root = data_path  # Since data_path is assumed to be image_root
            
            # 1. Identify and group files
            for filename in files:
                # Check for compatible extensions (case-insensitive) and existence of '_'
                if os.path.splitext(filename)[1].lower() in ('.jpg', '.jpeg', '.png') and '_' in filename:
                    try:
                        base_name_with_suffix, ext = os.path.splitext(filename)
                        if '_' in base_name_with_suffix:
                            base_name, suffix = base_name_with_suffix.rsplit('_', 1)
                            
                            if suffix in SUFFIXES:
                                # SIMPLIFIED: The relative path is just the filename
                                relative_path = filename
                                
                                if base_name not in grouped_files:
                                    grouped_files[base_name] = [None] * len(SUFFIXES) 
                                
                                index = SUFFIXES.index(suffix)
                                grouped_files[base_name][index] = relative_path
                    except Exception as e:
                        print(f"[Warning] Could not parse filename {filename}: {e}")
                        continue

            # 2. Check groups and create samples
            for base_name, path_list in grouped_files.items():
                if all(path is not None for path in path_list):
                    sample = {"image": path_list}
                    
                    # Check existence: root is data_path, path_list contains only filenames
                    # paths_full = [os.path.join(data_path, img) for img in path_list if img]
                    # To maintain consistency with the original code's variable access (self.script_args.image_root):
                    root_check = self.script_args.image_root
                    paths_full = [os.path.join(root_check, img) for img in path_list if img]
                    
                    if all(os.path.exists(p) for p in paths_full):
                        valid_samples.append(sample)
                    else:
                        print(f"[Warning] Skipping one folder sample due to missing file(s): {paths_full}")
                else:
                    print(f"[Warning] Skipping file group {base_name} due to missing required suffixes: {path_list}")
                    
        # ------------------ Original Logic: Handle File ------------------
        elif os.path.isfile(data_path):
            print(f"Detected file path. Loading samples from index file.")
            
            try:
                with open(data_path, "r") as f:
                    if data_path.endswith(".jsonl"):
                        data_list = [json.loads(line) for line in f]
                    elif data_path.endswith((".json", ".JSON")):
                        data_list = json.load(f)
                    else:
                        print(f"[Error] Unsupported file type: {data_path}. Must be .jsonl or .json")
                        return [] 
            except Exception as e:
                print(f"[Error] Could not load data from {data_path}: {e}")
                return [] 

            # Iterate through samples in the index file
            for ex in data_list:
                image_list = ex.get("image", [])
                if not image_list:
                    continue
                
                root = self.script_args.image_root
                # image_list contains paths relative to image_root (e.g., 00000/0_63_ref.jpg)
                paths = [os.path.join(root, img) for img in image_list if img]
                
                # Check if files exist
                if all(os.path.exists(p) for p in paths):
                    valid_samples.append(ex)
                else:
                    print(f"[Warning] Skipping one sample due to missing file(s): {paths}")
        
        # ------------------ Invalid Path ------------------
        else:
            print(f"[Error] Data path is neither a valid file nor a directory: {data_path}")

        print(f"Total samples loaded: {len(valid_samples)}")

        return valid_samples

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        example = self.samples[index]
        sample = {}

        image_list = example.get('image')
        if isinstance(image_list, list):
            if len(image_list) > 2:
                ref_rel, ori_rel, gt_rel = image_list[:3]
            elif len(image_list) > 1:
                ref_rel, ori_rel = image_list
                gt_rel = ori_rel
            else:
                ref_rel = ori_rel = gt_rel = image_list[0]
        else:
            ref_rel = ori_rel = gt_rel = image_list

        root = self.script_args.image_root

        def make_path(fname):
            return os.path.join(root, fname) if fname else None

        ref_fp = make_path(ref_rel)
        ori_fp = make_path(ori_rel)
        gt_fp  = make_path(gt_rel)

        # If any required image is missing, skip this sample entirely
        missing = []
        for role, path in [("ref_image", ref_fp), ("ori_image", ori_fp), ("gt_image", gt_fp)]:
            if not path or not os.path.exists(path):
                missing.append(f"{role}: {path}")

        if missing:
            print(f"[Warning] Skipping index {index} due to missing files: {', '.join(missing)}")
            return None
            # raise IndexError(f"Missing image(s) for sample {index}")
        
        # Otherwise, safely open images
        def safe_open_image(path):
            return Image.open(path).convert('RGB')

        sample['ref_image'] = safe_open_image(ref_fp)
        sample['ref_image_path'] = ref_fp

        sample['ori_image'] = safe_open_image(ori_fp)
        sample['ori_image_path'] = ori_fp

        sample['gt_image'] = safe_open_image(gt_fp)
        sample['gt_image_path'] = gt_fp

        sample['system_prompt']   = example.get('system_prompt', SYS)
        sample['custom_question'] = example.get('custom_question', MY_QUESTION)

        return sample


def score_reward(completions, solution, **kwargs):
    """
    For comparison tasks only:
      - Extract text from the <answer> tag.
      - If it exactly matches the solution (e.g., "Image A" or "Image B"), assign a reward of 1.0; otherwise, 0.0.
      - Preserve DEBUG logs by writing each match result to a file.
    """
    contents = [c[0]["content"] for c in completions]
    rewards = []
    answer_tag_pattern = r'<answer>(.*?)</answer>'

    for idx, (content, true_sol) in enumerate(zip(contents, solution)):
        reward = 0.0
        answer_text = ""
        try:
            m = re.search(answer_tag_pattern, content, re.DOTALL)
            if m:
                answer_text = m.group(1).strip()
                pat = re.compile(rf"^{re.escape(true_sol)}$")
                if pat.fullmatch(answer_text):
                    reward = 1.0
        except Exception as e:
            print(f"Error in computing comparison reward at idx {idx}:", e)

        rewards.append(reward)

        # DEBUG logging
        if os.getenv("DEBUG_MODE") == "true":
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available() and torch.distributed.is_initialized()
                else 0
            )
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            log_path = os.getenv("LOG_PATH", "comparison_reward.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"----- {now} Rank:{rank} Index:{idx} -----\n")
                f.write(f"Expected: {true_sol!r}\n")
                f.write(f"Answer:   {answer_text!r}\n")
                f.write(f"Content: {content}\n")
                f.write(f"Reward:   {reward}\n\n")

    return rewards


def format_reward(completions, **kwargs):
    """
    Reward = 1 if:
      - Completion contains a single-line <SoT>...<EoT> section.
      - The inside contains exactly 64 tokens of the form <MM\d+>.
    Otherwise, reward = 0.
    
    Returns: rewards
    """
    pattern = r"<SoT>([^\r\n]+?)<EoT>"

    completion_contents = [completion[0]["content"] for completion in completions]
    rewards, extracted_contents = [], []

    for content in completion_contents:
        # print(content)
        match = re.search(pattern, content)
        if not match:
            print(f"Match not found.")
            rewards.append(0.0)
            extracted_contents.append(None)
            continue

        inside_text = match.group(1)
        extracted_contents.append(inside_text)

        # Find all <MM123> tokens
        mm_ids = re.findall(r"<MM(\d+)>", inside_text)

        try:
            prediction_ids_flatten = np.array(list(map(int, mm_ids)))
        except ValueError:
            rewards.append(0.0)
            continue

        # Reward 1 only if exactly 64 tokens exist
        rewards.append(1.0 if len(prediction_ids_flatten) == 64 else 0.0)

    return rewards


def delta_E_reward(device="cuda"):
    
    print("Loading acetone reward model...")
    ckpt = torch.load(get_path("vq_ckpt_path", "PATH_TO_VQ_CKPT_PATH"), map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    vq_model = VQVAE3DLUT(
        codebook_size=ckpt.get("args", {}).get("codebook_size", 256),
        embedding_dim=ckpt.get("args", {}).get("embedding_dim", 64)
    )
    vq_model.load_state_dict(state_dict)
    vq_model.to(device)
    vq_model.eval()
    if os.getenv("DEBUG_MODE") == "true":
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            debug_mode = True
        else:
            debug_mode = False
    else:
        debug_mode = False

    def delta_E_score(prompts, completions, ori_image, gt_image, **kwargs):
        scores = []
        for i, output_text in enumerate(completions):
            try:
                with torch.no_grad():
                    prediction = output_text[0]['content'].split('<SoT>')[-1].split('<EoT>')[0]
                    prediction_ids_flatten = torch.tensor(np.array(list(map(int, re.findall(r"<MM(\d+)>", prediction)))))
                    prediction_ids = prediction_ids_flatten[:64].reshape(4,4,4).unsqueeze(0).to(device)
                    lut_pred = vq_model.decode_indices(prediction_ids)
                lut_pred_image = apply_lut(ori_image[i], lut_pred.squeeze(0))
                if lut_pred_image.size != gt_image[i].size:
                    print(f"Resizing image {i} from {lut_pred_image.size} to {gt_image[i].size}")
                    lut_pred_image = lut_pred_image.resize(gt_image[i].size)
                score = delta_e_between_images(lut_pred_image, gt_image[i])
                scores.append(score)
            except Exception as e:
                print("Error in computing reward:", e)
                scores.append(20)
        if debug_mode:
            print(f"delta_E: {scores}")
        return [ (1 / (max(score, 2) - 1)) for score in scores]
        # k = 0.5
        # tau = 2.0
        # return [2 / (1 + np.exp(k * (score - tau))) for score in scores]
    
    return delta_E_score

def color_similarity_reward(device="cuda"):
    
    print("Load color similarity assessment model...")
    ckpt = torch.load(get_path("vq_ckpt_path", "PATH_TO_VQ_CKPT_PATH"), map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    vq_model = VQVAE3DLUT(
        codebook_size=ckpt.get("args", {}).get("codebook_size", 256),
        embedding_dim=ckpt.get("args", {}).get("embedding_dim", 64)
    )
    vq_model.load_state_dict(state_dict)
    vq_model.to(device)
    vq_model.eval()

    if os.getenv("DEBUG_MODE") == "true":
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if rank == 0:
            debug_mode = True
        else:
            debug_mode = False
    else:
        debug_mode = False

    def color_similarity(prompts, completions, ori_image, ref_image, **kwargs):
        scores = []
        for i, output_text in enumerate(completions):
            try:
                with torch.no_grad():
                    prediction = output_text[0]['content'].split('<SoT>')[-1].split('<EoT>')[0]
                    prediction_ids_flatten = torch.tensor(np.array(list(map(int, re.findall(r"<MM(\d+)>", prediction)))))
                    prediction_ids = prediction_ids_flatten[:64].reshape(4,4,4).unsqueeze(0).to(device)
                    lut_pred = vq_model.decode_indices(prediction_ids)
                lut_pred_image = apply_lut(ori_image[i], lut_pred.squeeze(0))
                score = ColorSimilarity(lut_pred_image, ref_image[i])
                scores.append(score)
            except Exception as e:
                print("Error in computing reward:", e)
                scores.append(0.0)
        if debug_mode:
            print(f"color_similarity: {scores}")
        return scores
    
    return color_similarity

def aes_reward_remote(device="cuda"):
    """Use remote DeQA server and return a scoring function."""

    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    print("Load lut tokenizer...")
    ckpt = torch.load(get_path("vq_ckpt_path", "PATH_TO_VQ_CKPT_PATH"), map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    vq_model = VQVAE3DLUT(
        codebook_size=ckpt.get("args", {}).get("codebook_size", 256),
        embedding_dim=ckpt.get("args", {}).get("embedding_dim", 64)
    )
    vq_model.load_state_dict(state_dict)
    vq_model.to(device)
    vq_model.eval()

    url = "http://127.0.0.1:18087" # the deqa server address
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def deqa_score(prompts, completions, **kwargs):
        jpeg_images = []
        for i, output_text in enumerate(completions):
            try:
                with torch.no_grad():
                    prediction = output_text[0]['content'].split('<SoT>')[-1].split('<EoT>')[0]
                    prediction_ids_flatten = torch.tensor(np.array(list(map(int, re.findall(r"<MM(\d+)>", prediction)))))
                    prediction_ids = prediction_ids_flatten[:64].reshape(4,4,4).unsqueeze(0).to(device)
                    lut_pred = vq_model.decode_indices(prediction_ids)
                    lut_pred_image = apply_lut(ori_image[i], lut_pred.squeeze(0))
            except Exception as e:
                print("Error in computing reward:", e)
                # use a blank PIL image
                lut_pred_image = Image.new("RGB", (256, 256), (0, 0, 0))
                continue
            buffer = BytesIO()
            lut_pred_image.save(buffer, format="JPEG")
            jpeg_images.append(buffer.getvalue())

        data = {
            "images": jpeg_images
        }
        data_bytes = pickle.dumps(data)

        # send a request to the llava server
        response = sess.post(
        url,
        data=data_bytes,
        timeout=60,
        proxies={"http": None, "https": None}
        )
        response_data = pickle.loads(response.content)
        # print(response_data["scores"])

        all_scores = response_data["scores"]

        return all_scores
    
    return deqa_score

reward_funcs_registry = {
    "accuracy": score_reward,
    "format": format_reward,
}


def main(script_args, training_args, model_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reward_funcs = [delta_E_reward(device)]
    # reward_funcs = [delta_E_reward(device), format_reward]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LazyComparisonDataset(script_args)

    trainer_cls = AcetoneGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        import logging
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
