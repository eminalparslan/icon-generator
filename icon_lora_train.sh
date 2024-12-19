# export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export MODEL_NAME="kopyl/ui-icons-256"
export OUTPUT_DIR="./lora_checkpoints"
# export HUB_MODEL_ID="material-icon-lora"
export DATASET_PATH="./outlined_rasterized"

accelerate launch --mixed_precision="bf16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_PATH \
  --dataloader_num_workers=8 \
  --resolution=384 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="Heart icon, minimal, flat, simple, outlined, white background, few lines, four lines, 4 lines, 4 strokes" \
  --seed=1337

  # --validation_prompt="Simple minimalist outlined icon of right arrow, white background, black outline, vector style, clean lines, centered" \
  # --center_crop \
  # --random_flip \
  # --push_to_hub \
  # --hub_model_id=${HUB_MODEL_ID} \
