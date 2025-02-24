import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt

# Clear GPU memory cache
torch.cuda.empty_cache()

# Load Stable Diffusion model
model_id = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Use DPMSolverMultistepScheduler for better sampling
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Move pipeline to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Prompt for image generation
prompt = "a futuristic city with neon lights at night"
image = pipe(prompt, width=1000, height=1000).images[0]

# Display the image
plt.imshow(image)
plt.axis('off')  # Hide axis
plt.show()

# Save the image locally
image.save("generated_image.png")
