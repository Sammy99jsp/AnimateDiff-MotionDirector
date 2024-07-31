
DIFFUSERS_SDXL_UNET_CONFIG = {
    "act_fn": "silu",
    "addition_embed_type": "text_time",
    "addition_embed_type_num_heads": 64,
    "addition_time_embed_dim": 256,
    "attention_head_dim": [5, 10, 20],
    "block_out_channels": [320, 640, 1280],
    "center_input_sample": False,
    "class_embed_type": None,
    "class_embeddings_concat": False,
    "conv_in_kernel": 3,
    "conv_out_kernel": 3,
    "cross_attention_dim": 2048,
    "cross_attention_norm": None,
    "down_block_types": ["DownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D"],
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "encoder_hid_dim": None,
    "encoder_hid_dim_type": None,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 2,
    "mid_block_only_cross_attention": None,
    "mid_block_scale_factor": 1,
    "mid_block_type": "UNetMidBlock3DCrossAttn",
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "num_attention_heads": None,
    "num_class_embeds": None,
    "only_cross_attention": False,
    "out_channels": 4,
    "projection_class_embeddings_input_dim": 2816,
    "resnet_out_scale_factor": 1.0,
    "resnet_skip_time_act": False,
    "resnet_time_scale_shift": "default",
    "sample_size": 128,
    "time_cond_proj_dim": None,
    "time_embedding_act_fn": None,
    "time_embedding_dim": None,
    "time_embedding_type": "positional",
    "timestep_post_act": None,
    "transformer_layers_per_block": [1, 2, 10],
    "up_block_types": ["CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "UpBlock3D"],
    "upcast_attention": False,
    "use_linear_projection": True,
}


def convert_sdxl_unet_state_dict_to_diffusers(sd):
    unet_conversion_map = make_unet_conversion_map()

    conversion_dict = {sd: hf for sd, hf in unet_conversion_map}
    return convert_unet_state_dict(sd, conversion_dict)


def make_unet_conversion_map():
    unet_conversion_map_layer = []

    for i in range(3):  # num_blocks is 3 in sdxl
        # loop over downblocks/upblocks
        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                # no attention layers in down_blocks.3
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

            # if i > 0: commentout for sdxl
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # no downsample in down_blocks.3
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

            # no upsample in up_blocks.3
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3*i + 2}.{2}."  # change for sdxl
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2*j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    unet_conversion_map_resnet = [
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0.", "norm1."),
        ("in_layers.2.", "conv1."),
        ("out_layers.0.", "norm2."),
        ("out_layers.3.", "conv2."),
        ("emb_layers.1.", "time_emb_proj."),
        ("skip_connection.", "conv_shortcut."),
    ]

    unet_conversion_map = []
    for sd, hf in unet_conversion_map_layer:
        if "resnets" in hf:
            for sd_res, hf_res in unet_conversion_map_resnet:
                unet_conversion_map.append((sd + sd_res, hf + hf_res))
        else:
            unet_conversion_map.append((sd, hf))

    for j in range(2):
        hf_time_embed_prefix = f"time_embedding.linear_{j+1}."
        sd_time_embed_prefix = f"time_embed.{j*2}."
        unet_conversion_map.append((sd_time_embed_prefix, hf_time_embed_prefix))

    for j in range(2):
        hf_label_embed_prefix = f"add_embedding.linear_{j+1}."
        sd_label_embed_prefix = f"label_emb.0.{j*2}."
        unet_conversion_map.append((sd_label_embed_prefix, hf_label_embed_prefix))

    unet_conversion_map.append(("input_blocks.0.0.", "conv_in."))
    unet_conversion_map.append(("out.0.", "conv_norm_out."))
    unet_conversion_map.append(("out.2.", "conv_out."))

    return unet_conversion_map


def convert_unet_state_dict(src_sd, conversion_map):
    converted_sd = {}
    for src_key, value in src_sd.items():
        # さすがに全部回すのは時間がかかるので右から要素を削りつつprefixを探す
        src_key_fragments = src_key.split(".")[:-1]  # remove weight/bias
        while len(src_key_fragments) > 0:
            src_key_prefix = ".".join(src_key_fragments) + "."
            if src_key_prefix in conversion_map:
                converted_prefix = conversion_map[src_key_prefix]
                converted_key = converted_prefix + src_key[len(src_key_prefix) :]
                converted_sd[converted_key] = value
                break
            src_key_fragments.pop(-1)
        assert len(src_key_fragments) > 0, f"key {src_key} not found in conversion map"

    return converted_sd
