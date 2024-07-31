import os
import typing

from ..pipelines.pipeline_animation import AnimationPipeline
from ..pipelines.pipeline_xl import AnimationPipelineXL
import imageio
import numpy as np
from typing import Union

import torch
import torchvision
import torch.distributed as dist


from safetensors import safe_open
from tqdm import tqdm
from einops import rearrange
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from .convert_lora_safetensor_to_diffusers import convert_lora, load_diffusers_lora

from .xl import load_models_from_sdxl_checkpoint, MODEL_VERSION_SDXL_BASE_V1_0

def zero_rank_print(s):
    if not isinstance(s, str): s = repr(s)
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


class SD:
    @classmethod
    def load_weights(
        cls,
        pipeline: AnimationPipeline | AnimationPipelineXL,
        # Motion Modules
        motion_module_path: str = "",
        motion_module_lora_configs: list[typing.Any] = [],
        
        # Domain Adapter
        adapter_lora_path: str = "",
        adapter_lora_scale: float = 1.0,
        
        # Image Layers
        dreambooth_model_path: str = "",
        lora_model_path: str = "",
        lora_alpha                 = 0.8,
    ) -> None:
        r"""
        Loads the weights of the additional models into the pipeline.
        
        Args:
            pipeline (`AnimationPipeline`):
                AnimateDiff animation pipeline created from the `animatediff.pipelines.*` module classes.
            motion_module_path (`str` *optional*):
                Path to the pretrained motion module.
            motion_module_lora_configs (`list[...]` *optional*):
                Motion LoRAs to use.
            adapter_lora_path: (`str` *optional*):
                Path to the Domain Adapter LoRA.
            adapter_lora_scale: (`float` *optional*)
                Scale for the Domain Adapter LoRA.
                Default Value: `1.0`.   
            dreambooth_model_path (`str` *optional*):
                Path to a fine-tuned checkpoint.
            lora_model_path (`str` *optional*):
                Path to a SD LoRA.
            lora_alpha (`float`):
                TODO: ???
        """
        ...

class SD15(SD):
    @classmethod
    def load_weights(
        cls,
        pipeline: AnimationPipeline,
        # Motion Modules
        motion_module_path: str = "",
        motion_module_lora_configs: list[typing.Any] = [],
        
        # Domain Adapter
        adapter_lora_path: str = "",
        adapter_lora_scale: float = 1.0,
        
        # Image Layers
        dreambooth_model_path: str = "",
        lora_model_path: str = "",
        lora_alpha                 = 0.8,
    ) -> AnimationPipeline:
        # Load Motion Module
        unet_state_dict = {}
        if motion_module_path != "":
            print(f"load motion module from {motion_module_path}")
            motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
            motion_module_state_dict = motion_module_state_dict["state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
            unet_state_dict.update({name: param for name, param in motion_module_state_dict.items() if "motion_modules." in name})
            unet_state_dict.pop("animatediff_config", "")
        
        missing, unexpected = pipeline.unet.load_state_dict(unet_state_dict, strict=False)
        assert len(unexpected) == 0
        del unet_state_dict

        # Load SD1.5 checkpoint model
        if dreambooth_model_path != "":
            print(f"load dreambooth model from {dreambooth_model_path}")
            if dreambooth_model_path.endswith(".safetensors"):
                dreambooth_state_dict = {}
                with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f: # type: ignore
                    for key in f.keys():
                        dreambooth_state_dict[key] = f.get_tensor(key)
            elif dreambooth_model_path.endswith(".ckpt"):
                dreambooth_state_dict = torch.load(dreambooth_model_path, map_location="cpu")
                
            # 1. vae
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, pipeline.vae.config)
            pipeline.vae.load_state_dict(converted_vae_checkpoint)
            # 2. unet
            converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, pipeline.unet.config)
            pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
            # 3. text_model
            pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict)
            del dreambooth_state_dict
            
        # Load LoRA
        if lora_model_path != "":
            print(f"load lora model from {lora_model_path}")
            assert lora_model_path.endswith(".safetensors")
            lora_state_dict = {}
            with safe_open(lora_model_path, framework="pt", device="cpu") as f: # type: ignore
                for key in f.keys():
                    lora_state_dict[key] = f.get_tensor(key)
                    
            pipeline = convert_lora(pipeline, lora_state_dict, alpha=lora_alpha)
            del lora_state_dict

        # Load Domain Adapter LoRA
        if adapter_lora_path != "":
            print(f"load domain lora from {adapter_lora_path}")
            domain_lora_state_dict = torch.load(adapter_lora_path, map_location="cpu")
            domain_lora_state_dict = domain_lora_state_dict["state_dict"] if "state_dict" in domain_lora_state_dict else domain_lora_state_dict
            domain_lora_state_dict.pop("animatediff_config", "")

            pipeline = load_diffusers_lora(pipeline, domain_lora_state_dict, alpha=adapter_lora_scale)

        # Load Motion Module LoRA
        for motion_module_lora_config in motion_module_lora_configs:
            path, alpha = motion_module_lora_config["path"], motion_module_lora_config["alpha"]
            print(f"load motion LoRA from {path}")
            motion_lora_state_dict = torch.load(path, map_location="cpu")
            motion_lora_state_dict = motion_lora_state_dict["state_dict"] if "state_dict" in motion_lora_state_dict else motion_lora_state_dict
            motion_lora_state_dict.pop("animatediff_config", "")

            pipeline = load_diffusers_lora(pipeline, motion_lora_state_dict, alpha)

        return pipeline
        
class SDXL(SD):
    @classmethod
    def load_weights(
        cls,
        pipeline: AnimationPipelineXL,
        # Motion Modules
        motion_module_path: str = "",
        motion_module_lora_configs: list[typing.Any] = [],
        
        # Domain Adapter
        adapter_lora_path: str = "",
        adapter_lora_scale: float = 1.0,
        
        # Image Layers
        dreambooth_model_path: str = "",
        lora_model_path: str = "",
        lora_alpha                 = 0.8,
    ) -> AnimationPipelineXL:
        # Load SDXL checkpoint
        if dreambooth_model_path != "":
            (text_model1, text_model2, vae, unet, logit_scale, ckpt_info) = load_models_from_sdxl_checkpoint(
                model_version=MODEL_VERSION_SDXL_BASE_V1_0,
                ckpt_path=dreambooth_model_path,
                map_location='cpu'
            )
            
            unet_state_dict = unet.state_dict()
            pipeline.unet.load_state_dict(unet_state_dict, strict=False)
            pipeline.vae = vae if vae is not None else pipeline.vae
            pipeline.text_encoder = text_model1 if text_model1 is not None else pipeline.text_encoder
            pipeline.text_encoder_2 = text_model2 if text_model2 is not None else pipeline.text_encoder_2
            del unet
            del unet_state_dict
            del vae
            del text_model1
            del text_model2
            print(f'Loaded SDXL checkpoint from {dreambooth_model_path}')
        
        # Load Motion Module
        if motion_module_path != "":
            motion_module_ckpt = torch.load(motion_module_path, map_location='cpu')
            
            motion_module_state_dict = {}
            for k, v in motion_module_ckpt.items():
                if 'motion_module' in k and k in pipeline.unet.state_dict().keys():
                    motion_module_state_dict[k] = v
                elif 'motion_module' in k and k not in pipeline.unet.state_dict().keys():
                    print(k)

            pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
            del motion_module_ckpt
            del motion_module_state_dict
            print(f'Loaded motion module from {motion_module_path}...')
        
        # Load LoRA
        if lora_model_path != "":
            lora_state_dict = {}
            with safe_open(lora_model_path, framework='pt', device='cpu') as f: # type: ignore
                for k in f.keys():
                    lora_state_dict[k] = f.get_tensor(k)
            for k, v in lora_state_dict.items():
                if 'lora.up' in k:
                
                    down_key = k.replace('lora.up', 'lora.down')
                    if 'to_out' not in k:
                        original_key = k.replace('processor.', '').replace('_lora.up', '')
                    else:
                        original_key = k.replace('processor.', '').replace('_lora.up', '.0')
                    pipeline.unet.state_dict()[original_key] += lora_alpha * torch.mm(v, lora_state_dict[down_key])
            print(f'Loaded LoRA model from {lora_model_path}')

        # Load Domain Adapter LoRA
        if adapter_lora_path != "":
            print(f"load domain lora from {adapter_lora_path}")
            domain_lora_state_dict = torch.load(adapter_lora_path, map_location="cpu")
            domain_lora_state_dict = domain_lora_state_dict["state_dict"] if "state_dict" in domain_lora_state_dict else domain_lora_state_dict
            domain_lora_state_dict.pop("animatediff_config", "")

            pipeline = load_diffusers_lora(pipeline, domain_lora_state_dict, alpha=adapter_lora_scale)

        # Load Motion Module LoRA
        for motion_module_lora_config in motion_module_lora_configs:
            path, alpha = motion_module_lora_config["path"], motion_module_lora_config["alpha"]
            print(f"load motion LoRA from {path}")
            motion_lora_state_dict = torch.load(path, map_location="cpu")
            motion_lora_state_dict = motion_lora_state_dict["state_dict"] if "state_dict" in motion_lora_state_dict else motion_lora_state_dict
            motion_lora_state_dict.pop("animatediff_config", "")

            pipeline = load_diffusers_lora(pipeline, motion_lora_state_dict, alpha)

        return pipeline
