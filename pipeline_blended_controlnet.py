
from diffusers import DiffusionPipeline
from diffusers.loaders import LoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import EulerDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.image_processor import VaeImageProcessor

from transformers import CLIPTextModel, CLIPTokenizer

from PIL import Image

from typing import Type, Dict, List, Literal    

from utilities import Utilities, Layer

import torch

def limit(x: float, min: float, max: float) -> float:
    return x if x > min and x < max else min if x < min else max
    
def convert_to_pil_image(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images]

class BlendedControlNetPipeline(
    DiffusionPipeline,
    TextualInversionLoaderMixin,
    LoraLoaderMixin,
    FromSingleFileMixin,
    Utilities,
):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: DPMSolverMultistepScheduler
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        self.vae.enable_slicing()
        self.set_attention_slice("auto")

    def decode_latents(self, latents, generator):
        return self.vae.decode(
            latents / self.vae.config.scaling_factor,
            generator=generator
        ).sample

    def __call__(
        self,
        layers: List[Type[Layer]],
        controlnets: Dict[Literal['openpose', 'scribble'], Type[ControlNetModel]],
        generator: torch.Generator,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        lora_scale: float = 1.0,
        width: int = 1024,
        height: int = 1024,
        strength: float = 1.0,
        controlnet_conditioning_scale: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20
    ):
        # Prep vars
        device = self._execution_device
        guidance_scale = limit(guidance_scale, 1.0, 9.9)
        init_latents = None
        control_guidance_start = [control_guidance_start]
        control_guidance_end = [control_guidance_end]

        # Layer loop
        for layer_index, layer in enumerate(layers):

            print("Layer: " + str(layer_index) + "\n")

            # Encode input prompt
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                device=device,
                lora_scale=lora_scale,
                prompt_embeds=layer.prompt_embeds,
                negative_prompt_embeds=layer.negative_prompt_embeds
            )

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            # Prepare latent variables
            if layer_index == 0:
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
                self._num_timesteps = len(timesteps)

                init_latents = self.prepare_latents(
                    height=height,
                    width=width,
                    dtype=prompt_embeds.dtype,
                    device=device,
                    generator=generator
                )

                # BG-image to latent space
                latents = self.prepare_background_image(
                    image=Image.open("./background.png"),
                )
                
            # Switch up controlnets
            controlnet = controlnets[layer.controlnet_name]
        
            # Prepare mask
            mask = self.prepare_mask(
                mask=layer.mask,
                dest_size=(width // 8, height // 8),
                device=device
            )

            # Prepare controlnet_conditioning_image
            control_image = self.prepare_control_image(
                image=layer.control_image,
                width=width,
                height=height,
                device=device,
                dtype=controlnet.dtype
            )

            # Create tensor stating which controlnets to keep
            controlnet_keep = []
            for i in range(len(timesteps)):
                keeps = [
                    1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                    for s, e in zip(control_guidance_start, control_guidance_end)
                ]
                controlnet_keep.append(keeps[0])

            bg_latents = latents

            # Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(self.guidance_scale - 1)
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                ).to(device=device, dtype=latents.dtype)

            latents = init_latents

            # Denoising loop
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # Expand latents
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    
                    # Infer controlnet
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    down_block_res_samples, mid_block_res_sample = controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=control_image,
                        conditioning_scale=cond_scale,
                        guess_mode=False,
                        return_dict=False,
                    )

                    # Predict the noise residual
                    with torch.no_grad():
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=timestep_cond,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                            return_dict=False,
                        )[0]

                    # Peform Guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # Compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    # Blend layers according to mask
                    if layer_index < (num_inference_steps - 5):
                        # Add noise to bg
                        ennoised_bg = self.scheduler.add_noise(
                            bg_latents, init_latents, t
                        ).to(device)

                        # Mix latents
                        latents = latents * mask + ennoised_bg * (1 - mask)

                    progress_bar.update()

        # scale and decode the image latents with vae
        #  - - - - - - image = self.decode_latents(latents, generator)

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            image = self.vae.decode(latents, generator).sample

        # image = (image / 2 + 0.5).clamp(0, 1)
        # image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        # images = (image * 255).round().astype("uint8")

        # return images

        # Conver to pil and return
        return convert_to_pil_image(image)






