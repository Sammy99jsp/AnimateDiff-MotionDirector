import argparse
from dataclasses import dataclass
import datetime
import json
import os
from typing import Any, Literal, Optional
from omegaconf import DictConfig, ListConfig, OmegaConf

import torch
import torchvision.transforms as transforms

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, EulerDiscreteScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.xl.unet import UNet3DConditionModel as UNet3DConditionModelXL
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.pipelines.pipeline_xl import AnimationPipelineXL
from animatediff.utils.util import SD15, SDXL, save_videos_grid
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

from pathlib import Path
from PIL import Image
import numpy as np

@dataclass
class Args:
    pretrained_model_path = "models/StableDiffusion/stable-diffusion-v1-5"
    inference_config ="configs/inference/inference-v1.yaml"
    config: str
    L = 16 
    W = 512
    H = 512
    without_xformers = False
    
    @classmethod
    def parse(cls) -> "Args":
        parser = argparse.ArgumentParser()
        parser.add_argument("--pretrained-model-path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
        parser.add_argument("--inference-config",      type=str, default="configs/inference/inference-v1.yaml")    
        parser.add_argument("--config",                type=str, required=True)
        
        parser.add_argument("--L", type=int, default=16 )
        parser.add_argument("--W", type=int, default=512)
        parser.add_argument("--H", type=int, default=512)
        
        parser.add_argument("--without-xformers", action="store_true")

        args = parser.parse_args()
        self = Args(config=args.config)
        
        self.pretrained_model_path = args.pretrained_model_path or self.pretrained_model_path
        self.inference_config =args.inference_config or self.inference_config
        self.L = args.L or self.L
        self.W = args.W or self.W
        self.H = args.H or self.H
        self.without_xformers = args.without_xformers or self.without_xformers
        
        return self

def setup_controlnet(
    model_config: Any,
    savedir: str,
    unet:UNet3DConditionModel,
    vae: AutoencoderKL,
    length: int
) -> tuple[Optional[SparseControlNetModel], Optional[torch.Tensor]]:
    controlnet = controlnet_images = None
    if model_config.get("controlnet_path", "") != "":
        assert model_config.get("controlnet_images", "") != ""
        assert model_config.get("controlnet_config", "") != ""
        
        unet.config.num_attention_heads = 8                         # type: ignore
        unet.config.projection_class_embeddings_input_dim = None    # type: ignore

        controlnet_config = OmegaConf.load(model_config.controlnet_config)
        controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {})) # type: ignore

        print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
        controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")
        controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
        controlnet_state_dict.pop("animatediff_config", "")
        controlnet.load_state_dict(controlnet_state_dict)
        controlnet.cuda()

        image_paths = model_config.controlnet_images
        if isinstance(image_paths, str): image_paths = [image_paths]

        print(f"controlnet image paths:")
        for path in image_paths: print(path)
        assert len(image_paths) <= length

        image_transforms = transforms.Compose([
            transforms.RandomResizedCrop(
                (model_config.H, model_config.W), (1.0, 1.0), 
                ratio=(model_config.W/model_config.H, model_config.W/model_config.H)
            ),
            transforms.ToTensor(),
        ])

        if model_config.get("normalize_condition_images", False):
            def image_norm(image):
                image = image.mean(dim=0, keepdim=True).repeat(3,1,1)
                image -= image.min()
                image /= image.max()
                return image
        else: image_norm = lambda x: x # type: ignore
            
        controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]

        os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
        for i, image in enumerate(controlnet_images):
            Image.fromarray((255. * (image.numpy().transpose(1,2,0))).astype(np.uint8)).save(f"{savedir}/control_images/{i}.png")

        controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
        controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

        if controlnet.use_simplified_condition_embedding:
            num_controlnet_images = controlnet_images.shape[2]
            controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
            controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215 # type: ignore
            controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)
    
    return controlnet, controlnet_images

@dataclass
class PartialPipeline:
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    type: Literal["sd15"] | Literal["sdxl"]
    
    def __init__(self, pretrained_model_path: os.PathLike):
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").cuda() # type: ignore
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").cuda() # type: ignore
        self.type = "sd15"
    
    def build(
        self,
        unet: UNet3DConditionModel,
        inference_config: DictConfig,
        controlnet: Optional[SparseControlNetModel] = None
    ) -> AnimationPipeline | AnimationPipelineXL:
        return AnimationPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)), # type: ignore
        ).to("cuda")

@dataclass
class PartialPipelineXL(PartialPipeline):
    tokenizer_two: CLIPTokenizer
    text_encoder_two: CLIPTextModelWithProjection
    type: Literal["sd15"] | Literal["sdxl"]

    def __init__(self, pretrained_model_path: os.PathLike):
        super().__init__(pretrained_model_path)
        self.tokenizer_two = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer_2")
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder_2").cuda() # type: ignore
        self.type = "sdxl"

    def build(
        self,
        unet: UNet3DConditionModel,
        inference_config: DictConfig,
        controlnet: Optional[SparseControlNetModel] = None
    ) -> AnimationPipeline | AnimationPipelineXL:
        return AnimationPipelineXL(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            text_encoder_2=self.text_encoder_two,
            tokenizer_2=self.tokenizer_two,
            unet=unet,
            controlnet=controlnet,
            scheduler=EulerDiscreteScheduler(timestep_spacing='leading', steps_offset=1, **inference_config.noise_scheduler_kwargs), # type: ignore
        ).to("cuda")

class AD:
    type: Literal["sd15"] | Literal["sdxl"]
    args: Args
    config: ListConfig
    samples: list[torch.Tensor] = []
    savedir: str
    pipeline: PartialPipeline
    
    def __init__(self, args: Args):
        self.args = args
        
        # Init the output directory.
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.savedir = f"samples/{Path(args.config).stem}-{time_str}"
        os.makedirs(self.savedir)

        self.config: ListConfig = OmegaConf.load(args.config) # type: ignore
        self.samples = []
        
        pretrained_model_path = Path(args.pretrained_model_path).resolve()
        # Determine the model type.
        with open(pretrained_model_path.joinpath("model_index.json")) as model_index:
            meta = json.load(model_index)
            match meta["_class_name"]:
                case "StableDiffusionPipeline":
                    self.type = "sd15"
                case "StableDiffusionXLPipeline":
                    self.type = "sdxl"

        match self.type:
            case "sd15":
                self.pipeline = PartialPipeline(pretrained_model_path)
            case "sdxl":
                self.pipeline = PartialPipelineXL(pretrained_model_path)

    def save(self):
        r"""
        Save model outputs and parameters (e.g. seeds).
        """
        samples = torch.concat(self.samples)
        save_videos_grid(samples, os.path.join(self.savedir, "sample.gif"), n_rows=4)
        OmegaConf.save(self.config, os.path.join(self.savedir, "config.yaml"))
    
    def model_configs(self):
        for model_idx, model_config in enumerate(self.config):
            width = model_config.get("W", self.args.W)
            height = model_config.get("H", self.args.H)
            length = model_config.get("L", self.args.L)
            
            dimensions = (width, height, length)
            
            inference_config: DictConfig = OmegaConf.load(model_config.get("inference_config", self.args.inference_config)) # type: ignore

            unet: UNet3DConditionModel
            match self.type:
                case "sd15":
                    unet = UNet3DConditionModel.from_pretrained_2d(
                        self.args.pretrained_model_path,
                        subfolder="unet",
                        unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)
                    ).cuda()
                case "sdxl":
                    unet = UNet3DConditionModelXL.from_pretrained_2d(
                        self.args.pretrained_model_path,
                        subfolder="unet",
                        unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)
                    ).cuda()

            
            # load controlnet model
            controlnet, controlnet_images = setup_controlnet(model_config, self.savedir, unet, self.pipeline.vae, length)
            
            pipeline = self.pipeline.build(
                unet=unet,
                inference_config=inference_config,
                controlnet=controlnet,
            )
            
            match self.type:
                case "sd15":
                    pipeline = SD15.load_weights(
                        pipeline,                    # type: ignore
                        # motion module
                        motion_module_path         = model_config.get("motion_module", ""),
                        motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
                        # domain adapter
                        adapter_lora_path          = model_config.get("adapter_lora_path", ""),
                        adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
                        # image layers
                        dreambooth_model_path      = model_config.get("dreambooth_path", ""),
                        lora_model_path            = model_config.get("lora_model_path", ""),
                        lora_alpha                 = model_config.get("lora_alpha", 0.8),
                    ).to("cuda")
                case "sdxl":
                    pipeline = SDXL.load_weights(
                        pipeline,                    # type: ignore
                        # motion module
                        motion_module_path         = model_config.get("motion_module", ""),
                        motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
                        # domain adapter
                        adapter_lora_path          = model_config.get("adapter_lora_path", ""),
                        adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
                        # image layers
                        dreambooth_model_path      = model_config.get("dreambooth_path", ""),
                        lora_model_path            = model_config.get("lora_model_path", ""),
                        lora_alpha                 = model_config.get("lora_alpha", 0.8),
                    ).to("cuda")
            
            if is_xformers_available() and (not self.args.without_xformers):
                pipeline.unet.enable_xformers_memory_efficient_attention()
                if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()
    
            prompts      = model_config.prompt
            n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
            
            random_seeds = model_config.get("seed", [-1])
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
            
            self.config[model_idx].random_seed = []
            prompts = list(zip(prompts, n_prompts, random_seeds))
            
            yield model_config, pipeline, prompts, dimensions, controlnet_images
            
@torch.no_grad()
def main():
    sample_idx = 0
    ad = AD(Args.parse())
    savedir = ad.savedir
    samples = ad.samples
    
    for model_idx, (config, pipeline, prompts, (width, height, length), cnet_images) in enumerate(ad.model_configs()):
        for (prompt, n_prompt, random_seed) in prompts:
            
            # manually set random seed for reproduction
            if random_seed != -1: torch.manual_seed(random_seed)
            else: torch.seed()
            ad.config[model_idx].random_seed.append(torch.initial_seed())
            
            print(f"current seed: {torch.initial_seed()}")
            print(f"sampling {prompt} ...")
            sample: torch.Tensor = pipeline(
                prompt,
                negative_prompt     = n_prompt,
                num_inference_steps = config.steps,
                guidance_scale      = config.guidance_scale,
                width               = width,
                height              = height,
                video_length        = length,

                controlnet_images = cnet_images, # type: ignore
                controlnet_image_index = config.get("controlnet_image_indexs", [0]),
            ).videos # type: ignore -- return_dict=True
            samples.append(sample)

            prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
            save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
            print(f"save to {savedir}/sample/{prompt}.gif")
            
            sample_idx += 1

if __name__ == "__main__":
    main()
