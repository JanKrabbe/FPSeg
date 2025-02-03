# Code adapted for FPSeg

# Original file: segment-anything/build_sam.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import torch

from functools import partial

from .modeling import (
    FpsegMaskDecoder,
    FpsegPromptTuningEncoder,
    FpsegSam,
    ImageEncoderViT,
    TinyViT,
    TwoWayTransformer,
)


def build_fpseg_sam_vit_h(n_classes, prompt_size, checkpoint=None):
    return _build_fpseg_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        n_classes=n_classes,
        prompt_size=prompt_size,
        checkpoint=checkpoint,
    )


build_fpseg_sam = build_fpseg_sam_vit_h


def build_fpseg_sam_vit_l(n_classes, prompt_size, checkpoint=None):
    return _build_fpseg_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        n_classes=n_classes,
        prompt_size=prompt_size,
        checkpoint=checkpoint,
    )


def build_fpseg_sam_vit_b(n_classes, prompt_size, checkpoint=None):
    return _build_fpseg_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        n_classes=n_classes,
        prompt_size=prompt_size,
        checkpoint=checkpoint,
    )


def build_fpseg_mobilesam(n_classes, prompt_size, checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = FpsegSam(
        image_encoder=TinyViT(
            img_size=1024,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        ),
        prompt_encoder=FpsegPromptTuningEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            n_classes=n_classes,
            prompt_size=prompt_size,
        ),
        mask_decoder=FpsegMaskDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    mobile_sam.eval()
    if checkpoint is not None:
        load_pretrained_weights(mobile_sam, checkpoint)
    return mobile_sam


sam_model_registry = {
    "default": build_fpseg_sam_vit_h,
    "vit_h": build_fpseg_sam_vit_h,
    "vit_l": build_fpseg_sam_vit_l,
    "vit_b": build_fpseg_sam_vit_b,
    "mobilesam": build_fpseg_mobilesam,
}


def _build_fpseg_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    n_classes,
    prompt_size,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = FpsegSam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=FpsegPromptTuningEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            n_classes=n_classes,
            prompt_size=prompt_size,
        ),
        mask_decoder=FpsegMaskDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        load_pretrained_weights(sam, checkpoint)
    return sam


def load_pretrained_weights(
    model: FpsegSam,
    checkpoint: str,
) -> None:
    """
    Load pretrained weights. If the checkpoint is from SAM/MobileSAM only
    the corresponding weights are loaded.

    Arguments:
      model (FpsegSam): the FpsegSam model that is updated
      checkpoint (string): file path of the checkpoint
    """

    model_state_dict = model.state_dict()
    with open(checkpoint, "rb") as f:
        pretrained_state_dict = torch.load(f)

    # Check if pretrained encoder weights are for the right model
    if sum("image_encoder" in k for k in model_state_dict) != sum(
        "image_encoder" in k for k in pretrained_state_dict
    ):
        raise RuntimeError("The provided checkpoint does not contain the selected image encoder.")

    pretrained_state_dict = {
        k: v for k, v in pretrained_state_dict.items() if k in model_state_dict
    }

    # Handle mask_tokens separately because their dimension changed
    pretrained_state_dict["mask_decoder.mask_tokens.weight"] = pretrained_state_dict[
        "mask_decoder.mask_tokens.weight"
    ][0:1]

    model_state_dict.update(pretrained_state_dict)

    model.load_state_dict(model_state_dict)
