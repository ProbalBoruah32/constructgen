import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import requests
import json

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Clear CUDA cache
torch.cuda.empty_cache()

# Load Stable Diffusion model
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipe.scheduler = DPMSolverMultistepScheduler
