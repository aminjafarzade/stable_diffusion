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


prompt = "friends playing football in Istanbul Airport. Ultra Realistic"
uncond_prompt = ''
do_cfg = True
cfg_scale = 10


# IMAGE to IMAGE
input_image = None
image_path = '../images/dog.jpg'
# input_image = Image.open(image_path)
strength = 0.8

sampler = 'ddim'
num_inference_steps = 60
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

img = Image.fromarray(output_image)
img.save('../images/football.jpg')

