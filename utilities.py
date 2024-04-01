from diffusers.utils import scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import randn_tensor

from typing import Type, Literal, Tuple, Optional
from PIL import Image
import numpy as np

import torch

class Layer:
    def __init__(
        self,
        prompt_embeds: Type[torch.FloatTensor],
        negative_prompt_embeds: Type[torch.FloatTensor],
        control_image: Optional[Type[Image.Image]] = None,
        mask: Optional[Type[Image.Image]] = None,
        controlnet_name: Optional[Literal['openpose', 'scribble']] = None
    ) -> None:
        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds

        if control_image is not None:
            self.control_image = control_image

        if mask is not None:
            self.mask = mask

        if controlnet_name is not None:
            self.controlnet_name = controlnet_name

class Utilities:

    def encode_prompt(
        self,
        device: str,
        prompt_embeds: Type[torch.FloatTensor],
        negative_prompt_embeds: Type[torch.FloatTensor],
        lora_scale: float = 1.0,
    ):
        self._lora_scale = lora_scale
        scale_lora_layers(self.text_encoder, lora_scale)

        # Enforce 1 batch size and 1 image per prompt
        # hard code once everything is working
        # batch_size = 1
        # num_images_per_prompt = 1

        prompt_embeds_dtype = self.text_encoder.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
        
        # negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        negative_prompt_embeds = negative_prompt_embeds.view(1, seq_len, -1)

        unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds
    
    def prepare_background_image(
        self,
        image: Type[Image.Image],
    ):
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to("cuda")
        image = image.half()
        latents = self.vae.encode(image)["latent_dist"].mean
        latents = latents * 0.18215

        return latents
    
    def prepare_control_image(
        self,
        image: Type[Image.Image],
        width: int,
        height: int,
        device: int,
        dtype,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)

        image = image.to(device=device, dtype=dtype)

        image = torch.cat([image] * 2)

        return image
    
    def prepare_mask(
        self,
        mask: Type[Image.Image],
        dest_size: Tuple[float, float],
        device: str
    ):
        mask = mask.convert("L")
        mask = mask.resize(dest_size, Image.NEAREST)
        mask = np.array(mask) / 255
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask[np.newaxis, np.newaxis, ...]
        return torch.from_numpy(mask).half().to(device)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(
        self,
        num_inference_steps: int,
        strength: float,
        device: str
    ):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start
    
    def prepare_latents(
        self,
        height: int,
        width: int,
        dtype,
        device: str,
        generator
    ):
        num_channels_latents = self.unet.config.in_channels

        shape = (1, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
    
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
