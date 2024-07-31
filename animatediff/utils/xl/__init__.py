import os
from typing import Any, Optional, Tuple
from packaging import version
import torch
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors.torch import load_file
import torch.types
import transformers
from transformers import CLIPTextModel, CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import AutoencoderKL

from .lora import SdxlUNet2DConditionModel
from .vae import convert_ldm_vae_checkpoint, create_vae_diffusers_config
from .unet import DIFFUSERS_SDXL_UNET_CONFIG, convert_sdxl_unet_state_dict_to_diffusers

MODEL_VERSION_SDXL_BASE_V1_0 = "sdxl_base_v1-0"

def load_models_from_sdxl_checkpoint(
    model_version: str,
    ckpt_path: os.PathLike | str,
    map_location,
    dtype=None,
    unet_only=False
) -> Tuple[
    Optional[CLIPTextModel],
    Optional[CLIPTextModelWithProjection],
    Optional[AutoencoderKL],
    Any,
    Any,
    Optional[Tuple[int, Optional[int]]]]:
    r"""
    Load the constituent models from an SDXL checkpoint.
    
    Args:
        model_version (`str`):
            SDXL model version (usually `"sdxl_base_v1-0"`) -- reserved for future use.
        ckpt_path (`os.PathLike | str`):
            Path to the SDXL checkpoint.
        map_location (`str | int`):
            Torch device to load the tensors onto.
        dtype (`Optional[str]` *optional*):
            Reserved for full_fp16/bf16 precision integration.
            Text Encoder will remain fp32, because it runs on CPU when caching.
        unet_only (`bool` *optional*):
            Only load the UNet from this checkpoint.
            Default: `False`.
    """
    
    # model_version is reserved for future use
    # dtype is reserved for full_fp16/bf16 integration. Text Encoder will remain fp32, because it runs on CPU when caching

    # Load the state dict
    if is_safetensors(ckpt_path):
        checkpoint = None
        try:
            state_dict = load_file(ckpt_path, device=map_location)
        except:
            state_dict = load_file(ckpt_path)  # prevent device invalid Error
        epoch = None
        global_step = None
    else:
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("global_step", 0)
        else:
            state_dict = checkpoint
            epoch = 0
            global_step = 0
        checkpoint = None

    # U-Net
    print("building U-Net")
    with init_empty_weights():
        unet = SdxlUNet2DConditionModel()

    print("loading U-Net from checkpoint")
    unet_sd = {}
    for k in list(state_dict.keys()):
        if k.startswith("model.diffusion_model."):
            unet_sd[k.replace("model.diffusion_model.", "")] = state_dict.pop(k)
    info = _load_state_dict_on_device(unet, unet_sd, device=map_location)
    print("U-Net: ", info)
    unet_sd= unet.state_dict()
    du_unet_sd = convert_sdxl_unet_state_dict_to_diffusers(unet_sd)
    
    from ...models.xl.unet import UNet3DConditionModel
    diffusers_unet = UNet3DConditionModel(**DIFFUSERS_SDXL_UNET_CONFIG)
    diffusers_unet.load_state_dict(du_unet_sd)

    if unet_only:
        return None, None, None, diffusers_unet, None, None

    # Text Encoders
    print("building text encoders")

    # Text Encoder 1 is same to Stability AI's SDXL
    text_model1_cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=768,
        # torch_dtype="float32",
        # transformers_version="4.25.0.dev0",
    )
    text_model1 = CLIPTextModel._from_config(text_model1_cfg)

    # Text Encoder 2 is different from Stability AI's SDXL. SDXL uses open clip, but we use the model from HuggingFace.
    # Note: Tokenizer from HuggingFace is different from SDXL. We must use open clip's tokenizer.
    text_model2_cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=1280,
        intermediate_size=5120,
        num_hidden_layers=32,
        num_attention_heads=20,
        max_position_embeddings=77,
        hidden_act="gelu",
        layer_norm_eps=1e-05,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        model_type="clip_text_model",
        projection_dim=1280,
        # torch_dtype="float32",
        # transformers_version="4.25.0.dev0",
    )
    text_model2 = CLIPTextModelWithProjection(text_model2_cfg)

    print("loading text encoders from checkpoint")
    te1_sd = {}
    te2_sd = {}
    for k in list(state_dict.keys()):
        if k.startswith("conditioner.embedders.0.transformer."):
            te1_sd[k.replace("conditioner.embedders.0.transformer.", "")] = state_dict.pop(k)
        elif k.startswith("conditioner.embedders.1.model."):
            te2_sd[k] = state_dict.pop(k)

    if version.parse(transformers.__version__) > version.parse("4.31"):
        # After transformers==4.31, position_ids becomes a persistent=False buffer (so we musn't supply it)
        # https://github.com/huggingface/transformers/pull/24505
        # https://github.com/mlfoundations/open_clip/pull/595
        if 'text_model.embeddings.position_ids' in te1_sd:
            del te1_sd['text_model.embeddings.position_ids']
    info1 = text_model1.load_state_dict(te1_sd)
    print("text encoder 1:", info1)

    converted_sd, logit_scale = convert_sdxl_text_encoder_2_checkpoint(te2_sd, max_length=77)
    info2 = text_model2.load_state_dict(converted_sd)
    print("text encoder 2:", info2)

    # prepare vae
    print("building VAE")
    vae_config = create_vae_diffusers_config()
    vae = AutoencoderKL(**vae_config)  # .to(device)

    print("loading VAE from checkpoint")
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)
    info = vae.load_state_dict(converted_vae_checkpoint)
    print("VAE:", info)

    ckpt_info = (epoch, global_step) if epoch is not None else None
    return text_model1, text_model2, vae, diffusers_unet, logit_scale, ckpt_info


def is_safetensors(path):
    return os.path.splitext(path)[1].lower() == ".safetensors"

def _load_state_dict_on_device(model, state_dict, device, dtype=None):
    """
    Load `state_dict` without allocating new tensors.
    """
    # dtype will use fp32 as default
    missing_keys = list(model.state_dict().keys() - state_dict.keys())
    unexpected_keys = list(state_dict.keys() - model.state_dict().keys())

    # similar to model.load_state_dict()
    if not missing_keys and not unexpected_keys:
        for k in list(state_dict.keys()):
            set_module_tensor_to_device(model, k, device, value=state_dict.pop(k), dtype=dtype)
        return "<All keys matched successfully>"

    # error_msgs
    error_msgs: list[str] = []
    if missing_keys:
        error_msgs.insert(0, "Missing key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in missing_keys)))
    if unexpected_keys:
        error_msgs.insert(0, "Unexpected key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in unexpected_keys)))

    raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(model.__class__.__name__, "\n\t".join(error_msgs)))



def convert_sdxl_text_encoder_2_checkpoint(checkpoint, max_length):
    SDXL_KEY_PREFIX = "conditioner.embedders.1.model."

    # SD2のと、基本的には同じ。logit_scaleを後で使うので、それを追加で返す
    # logit_scaleはcheckpointの保存時に使用する
    def convert_key(key):
        # common conversion
        key = key.replace(SDXL_KEY_PREFIX + "transformer.", "text_model.encoder.")
        key = key.replace(SDXL_KEY_PREFIX, "text_model.")

        if "resblocks" in key:
            # resblocks conversion
            key = key.replace(".resblocks.", ".layers.")
            if ".ln_" in key:
                key = key.replace(".ln_", ".layer_norm")
            elif ".mlp." in key:
                key = key.replace(".c_fc.", ".fc1.")
                key = key.replace(".c_proj.", ".fc2.")
            elif ".attn.out_proj" in key:
                key = key.replace(".attn.out_proj.", ".self_attn.out_proj.")
            elif ".attn.in_proj" in key:
                key = None  # 特殊なので後で処理する
            else:
                raise ValueError(f"unexpected key in SD: {key}")
        elif ".positional_embedding" in key:
            key = key.replace(".positional_embedding", ".embeddings.position_embedding.weight")
        elif ".text_projection" in key:
            key = key.replace("text_model.text_projection", "text_projection.weight")
        elif ".logit_scale" in key:
            key = None  # 後で処理する
        elif ".token_embedding" in key:
            key = key.replace(".token_embedding.weight", ".embeddings.token_embedding.weight")
        elif ".ln_final" in key:
            key = key.replace(".ln_final", ".final_layer_norm")
        # ckpt from comfy has this key: text_model.encoder.text_model.embeddings.position_ids
        elif ".embeddings.position_ids" in key:
            key = None  # remove this key: make position_ids by ourselves
        return key

    keys = list(checkpoint.keys())
    new_sd = {}
    for key in keys:
        new_key = convert_key(key)
        if new_key is None:
            continue
        new_sd[new_key] = checkpoint[key]

    # attnの変換
    for key in keys:
        if ".resblocks" in key and ".attn.in_proj_" in key:
            # 三つに分割
            values = torch.chunk(checkpoint[key], 3)

            key_suffix = ".weight" if "weight" in key else ".bias"
            key_pfx = key.replace(SDXL_KEY_PREFIX + "transformer.resblocks.", "text_model.encoder.layers.")
            key_pfx = key_pfx.replace("_weight", "")
            key_pfx = key_pfx.replace("_bias", "")
            key_pfx = key_pfx.replace(".attn.in_proj", ".self_attn.")
            new_sd[key_pfx + "q_proj" + key_suffix] = values[0]
            new_sd[key_pfx + "k_proj" + key_suffix] = values[1]
            new_sd[key_pfx + "v_proj" + key_suffix] = values[2]

    # Create position_ids only for *old* transformers versions.
    # After transformers==4.31, position_ids becomes a persistent=False buffer (so we musn't supply it)
    # https://github.com/huggingface/transformers/pull/24505
    # https://github.com/mlfoundations/open_clip/pull/595
    if version.parse(transformers.__version__) <= version.parse("4.31"):
        # original SD にはないので、position_idsを追加
        position_ids = torch.Tensor([list(range(max_length))]).to(torch.int64)
        new_sd["text_model.embeddings.position_ids"] = position_ids

    # logit_scale はDiffusersには含まれないが、保存時に戻したいので別途返す
    logit_scale = checkpoint.get(SDXL_KEY_PREFIX + "logit_scale", None)

    return new_sd, logit_scale
