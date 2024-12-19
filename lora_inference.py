from diffusers import AutoPipelineForText2Image
import torch

subject = "Thermometer"

# model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
model = "kopyl/ui-icons-256"

pipeline = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch.float16).to("cuda:1")
pipeline.load_lora_weights("./lora_checkpoints/checkpoint-15000", weight_name="pytorch_lora_weights.safetensors")

image = pipeline(f"{subject} icon, minimal, flat, simple, outlined, white background, few lines, four lines, 4 lines, 4 strokes").images[0]
image.save(f"inference/{subject}.png")
