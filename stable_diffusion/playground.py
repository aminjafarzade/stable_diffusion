import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch


DEVICE = 'cpu'
ALLOW_CUDA = False
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = 'cuda'
# elif (torch.has_mps or torch.backend.mps.is_available()) and ALLOW_MPS:
#   DEVICE = 'mps'

print(f"Using {DEVICE}")

tokenizer = CLIPTokenizer("../data/vocab.json",
                          merges_file="../data/merges.txt")
model_file = '../data/v1-5-pruned-emaonly.ckpt'
models = model_loader.preload_weights_from_standard_weights(model_file, DEVICE)


# TEXT to IMAGE
prompt = "Generate a realistic 8K resolution of given image of a young man happily posing, and beside him, add a beautiful and smiling young woman as his girlfriend. Ensure the scene is natural, with appropriate lighting, shadows, and facial expressions reflecting a genuine and happy moment between the couple"
uncond_prompt = ''
do_cfg = True
cfg_scale = 7


# IMAGE to IMAGE
input_image = None
image_path = '../images/amin.jpg'
input_image = Image.open(image_path)
strength = 0.7

sampler = 'ddpm'
num_inference_steps = 50
seed = 42

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)

Image.fromarray(output_image)
