# In-built
import platform
import time
import base64
from io import BytesIO
import datetime

# Packages
import torch
import requests
from flask import Flask, request
from flask_restful import Resource, Api
from PIL import Image
from diffusers import ControlNetModel
from compel import Compel
from RealESRGAN import RealESRGAN

# Local
from pipeline_blended_controlnet import BlendedControlNetPipeline
from utilities import Layer
from enhance import enhance

# See: https://huggingface.co/docs/diffusers/en/optimization/fp16
torch.backends.cuda.matmul.allow_tf32 = True

def saveBytescale (data):
    headers = {
        'Authorization': 'Bearer public_12a1yrrGGApHW4eVGAfq3RnXk9uv',
        'Content-Type': 'image/png',
    }
    return requests.post('https://api.bytescale.com/v2/accounts/12a1yrr/uploads/binary', headers=headers, data=data)

def decode_base64_image(image_string):
    image_string = image_string[len("data:image/png;base64,"):]
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    image = Image.open(buffer)
    rgb = image.convert('RGB')
    return rgb

def generate():
    ## Local development -- Mac doesn't like float16
    if platform.system() == 'Darwin':
        device = "mps"
        torch_dtype = torch.float32
    else:
        device = "cuda"
        torch_dtype = torch.float16

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

    ## Text Embeddings
    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
    prompt_01 = "young man turning around, suprised, (sketch artstyle)---, graycale, monochrome, solo"
    prompt_embeds_01 = compel(prompt_01)
    negative_prompt = "glasses----, (extra fingers), (fewer fingers), (low quality), (worst quality), (bad anatomy), (inaccurate limb), (ugly eyes), (inaccurate eyes), (extra digit), (fewer digits), (extra arms), (extra navel), blurred, (out of focus), soft, deformed, watermark, (movie poster), (large breasts), (censored), (mosaic censoring), (piercings), (multiple), error, cropped, (low res), artifacts, (compression artifacts), ugly, duplicate, morbid, mutilated, disfigured, gross, malformed, missing, username, signature, faded" 
    negative_prompt_embeds = compel(negative_prompt)

    ## Layers
    layers = [
        Layer(
            prompt_embeds=prompt_embeds_01,
            negative_prompt_embeds=negative_prompt_embeds,
            control_image=Image.open("images/control_04.png"),
            mask=Image.open("images/mask_01.png"),
            controlnet_name="openpose"
        )
    ]

    # Make Prediction
    generator = torch.Generator(device=device).manual_seed(2)

    image = pipe(
        layers=layers,
        controlnets={
            "scribble" : scribble,
            "openpose" : openpose
        },
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

    return image

app = Flask(__name__)
api = Api(app)

class Predict(Resource):
    def post(self):
        time_start = time.time()

        req = request.json
        layers=req.get("layers")

        # ignore layers for now, quick test
        image = generate()

        with BytesIO() as image_binary:
            image.save(image_binary, "png")
            image_binary.seek(0)
            result = saveBytescale(image_binary)

        # Show total time
        time_end = time.time()
        print("Total time:", time_end - time_start, "s")

        return result.json()

api.add_resource(Predict, "/")

# # Settings
# model = "queratograySketch_v10.safetensors"
# lora = "none"

# # Pipe
# pipe = BlendedControlNetPipeline.from_single_file(
#     "./models/" + model,
#     use_safetensors=True
# ).to(device)

# # Lora
# # pipe.load_lora_weights("./models/" + lora)

# # Control nets
# scribble = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble").to(device)
# openpose = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose").to(device)

# # Text embeddings
# compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
# prompt_01 = "young man walking, (sketch artstyle)---, graycale, monochrome, solo"
# prompt_embeds_01 = compel(prompt_01)
# negative_prompt = "glasses----, (extra fingers), (fewer fingers), (low quality), (worst quality), (bad anatomy), (inaccurate limb), (ugly eyes), (inaccurate eyes), (extra digit), (fewer digits), (extra arms), (extra navel), blurred, (out of focus), soft, deformed, watermark, (movie poster), (large breasts), (censored), (mosaic censoring), (piercings), (multiple), error, cropped, (low res), artifacts, (compression artifacts), ugly, duplicate, morbid, mutilated, disfigured, gross, malformed, missing, username, signature, faded" 
# negative_prompt_embeds = compel(negative_prompt)
# prompt_02 = "man with a beard sitting in a pub, (sketch artstyle), graycale--, solo, masterpiece"
# prompt_embeds_02 = compel(prompt_02)

# # Layers
# layers = [
#     Layer(
#         prompt_embeds=prompt_embeds_02,
#         negative_prompt_embeds=negative_prompt_embeds,
#         control_image=Image.open("images/control_04.png"),
#         mask=Image.open("images/mask_01.png"),
#         controlnet_name="openpose"
#     )
# ]

# # Make prediction
# generator = torch.Generator(device=device).manual_seed(2)

# image = pipe(
#     layers=layers,
#     controlnets={
#         "scribble" : scribble,
#         "openpose" : openpose
#     },
#     generator=generator,
#     lora_scale=0.2,
#     guidance_scale=9.0,
#     controlnet_conditioning_scale=0.85
# )[0]
