#!/bin/bash
# export WANDB_PROJECT=AcetoneVLM
export WANDB_MODE=offline
# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=23333
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=scripts/zero2.json

# Model configuration
llm=outputs/pretrained/qwen2.5-vl-3b

# Training hyperparameters
lr=1e-4
# lr=5e-5
batch_size=16
grad_accum_steps=4

# Training entry point
entry_file=pipeline/acetone_train.py

# Dataset configuration (replace with public dataset names)
datasets=acetone

# Output configuration
run_name="acetoneVLM-SFT-augX"
output_dir=outputs/ckpt/vlm-c1024-test

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 2 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1500 \
    --save_total_limit 100 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.1 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}