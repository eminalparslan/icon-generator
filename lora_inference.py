from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda:1")
pipeline.load_lora_weights("./diffusers/examples/text_to_image/output/checkpoint-1000", weight_name="pytorch_lora_weights.safetensors")
image = pipeline("right arrow").images[0]
