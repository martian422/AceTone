set -x

export DEBUG_MODE=false
export WANDB_PROJECT=AcetoneGRPO
# export WANDB_MODE=offline
RUN_NAME="acetone-dE-newbase-pref-8k-data"
export LOG_PATH="outputs/r1-debug/debug_log_$RUN_NAME.txt"

NCCL_DEBUG=WARN

MODEL_NAME_OR_PATH="${ACETONE_MODEL_NAME_OR_PATH:-PATH_TO_ACETONE_MODEL_DIR}"
DATASET="${ACETONE_GRPO_DATASET_JSONL:-outputs/workspace/acetone_dataset/annotations/rl-8k.jsonl}"
IMAGE_FOLDER="${ACETONE_GRPO_IMAGE_ROOT:-outputs/workspace/acetone_dataset/images/rl-8k}"

torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=23333 \
    grpo/acetone_r1.py \
    --deepspeed grpo/local_scripts/zero2.json \
    --output_dir outputs/ckpt/grpo/$RUN_NAME \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dataset $DATASET \
    --image_root $IMAGE_FOLDER \
    --max_prompt_length 2048 \
    --num_generations 8 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 1000 \
    --save_only_model false \
    --score_reward_threshold 1.0 \
    --use_pref true \
    --beta 0.01
