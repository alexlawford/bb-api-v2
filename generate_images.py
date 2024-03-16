# In Built
import platform
from io import BytesIO
import base64

# Packages
import torch
from diffusers import ControlNetModel
from compel import Compel
from RealESRGAN import RealESRGAN
from PIL import Image

# Local
from pipeline_blended_controlnet import BlendedControlNetPipeline
from utilities import Layer
from enhance import enhance

# See: https://huggingface.co/docs/diffusers/en/optimization/fp16
torch.backends.cuda.matmul.allow_tf32 = True

def decode_base64_image(image_string):
    image_string = image_string[len("data:image/png;base64,"):]
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    image = Image.open(buffer)
    rgb = image.convert('RGB')
    return rgb

def generate_prompt_embeds(prompt, negative_prompt, pipe):
    standard_positive = " (high quality), (best quality), masterpiece"
    standard_negative = " glasses----, (extra fingers), (fewer fingers), (low quality), (worst quality), (bad anatomy), (inaccurate limb), (ugly eyes), (inaccurate eyes), (extra digit), (fewer digits), (extra arms), (extra navel), blurred, (out of focus), soft, deformed, watermark, (movie poster), (large breasts), (censored), (mosaic censoring), (piercings), (multiple), error, cropped, (low res), artifacts, (compression artifacts), ugly, duplicate, morbid, mutilated, disfigured, gross, malformed, missing, username, signature, faded"
    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
    return compel(prompt + standard_positive), compel(negative_prompt + standard_negative)

def execute_pipeline(layers, pipe, controlnets, device):
    generator = torch.Generator(device=device).manual_seed(2)

    image = pipe(
        layers=layers,
        controlnets=controlnets,
        generator=generator,
        lora_scale=0.2,
        guidance_scale=9.0,
        controlnet_conditioning_scale=0.85
    )[0]

    # Upscale & Enhance
    upscaler = RealESRGAN(torch.device(device), scale=2)
    upscaler.load_weights('weights/RealESRGAN_x2.pth', download=True)
    scaled = upscaler.predict(image)
    enhanced = enhance(scaled)
    resized = enhanced.resize((512,512))

    return resized

def setup_pipe(device, torch_dtype):
    ## Weights
    checkpoint = "queratograySketch_v10.safetensors"

    ## Pipe
    pipe = BlendedControlNetPipeline.from_single_file(
        "./weights/" + checkpoint,
        torch_dtype=torch_dtype,
        use_safetensors=True
    ).to(device)

    # Control Nets
    scribble = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch_dtype).to(device)
    openpose = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch_dtype).to(device)

    return pipe, {"scribble" : scribble, "openpose" : openpose}


def generate(layers_raw):
    pipe, controlnets = setup_pipe("cuda", torch.float16)

    layers = []

    for layer in layers_raw:

        prompt_embeds, negative_prompt_embeds = generate_prompt_embeds(layer["prompt"], layer["negative prompt"], pipe)

        l = Layer(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            control_image=decode_base64_image(layer["control"]),
            mask=Image.open(layer["mask"]),
            controlnet_name=layer["type"]
        )
        layers.append(l)
    
    return execute_pipeline(layers, pipe, controlnets, "cuda")

# # Settings
# lora = "none"

# # Pipe
# pipe = BlendedControlNetPipeline.from_single_file(
#     "./models/" + model,
#     use_safetensors=True
# ).to(device)

# # Lora
# # pipe.load_lora_weights("./models/" + lora)
# image = pipe(
#     lora_scale=0.2,
# )[0]
