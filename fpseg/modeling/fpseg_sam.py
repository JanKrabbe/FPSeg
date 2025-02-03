# Code adapted for FPSeg

# Original file: segment-anything/modeling/sam.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple, Union

from .fpseg_mask_decoder import FpsegMaskDecoder
from .fpseg_prompt_tuning_encoder import FpsegPromptTuningEncoder
from .image_encoder import ImageEncoderViT
from .tiny_vit_sam import TinyViT


class FpsegSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: Union[ImageEncoderViT, TinyViT],
        prompt_encoder: FpsegPromptTuningEncoder,
        mask_decoder: FpsegMaskDecoder,
        original_size: Tuple[int, int] = (1024, 2048),
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        FpsegSAM predicts class masks from an image and tuned prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (FpsegPromptTuningEncoder): Holds (tuned)
            prompt tokens. Named as in SAM to remain compatibility with pretrained weights.
          mask_decoder (FpsegMaskDecoder): Predicts masks from the image embeddings
            and prompt tokens. Named as in SAM to remain compatibility with pretrained weights.
          original_image_size (tuple(int, int)): The size of the image used to calculate the
             embedding before transformation, in (H, W) format. Determines size of output mask.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.original_size = original_size
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_images: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images. One mask is predicted
        for each class, that is specified by a set of prompt tokens in the
        FpsegPromptTuningEncoder.

        Arguments:
          batched_images (torch.Tensor): Batch of input images, with shape
            Bx3xHxW, where b is the batch size.

        Returns:
          (torch.Tensor): A tensor of masks with shape Bxn_classesxHxW, where
            B is the batch size and (H, W) is the original size of the image.
        """

        input_images = self.preprocess(batched_images)
        image_embeddings = self.image_encoder(input_images)

        prompt_embeddings, dense_embeddings = self.prompt_encoder()
        n_classes = int(prompt_embeddings.shape[0])

        outputs = []
        for curr_embedding in image_embeddings:
            expanded_curr_embedding = curr_embedding.unsqueeze(0).repeat_interleave(n_classes, 0)

            low_res_masks = self.mask_decoder(
                image_embeddings=expanded_curr_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=prompt_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=batched_images.shape[-2:],
                original_size=self.original_size,
            )
            masks = masks > self.mask_threshold

            outputs.append(masks)
        return torch.stack(outputs, dim=0)

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in n_classesxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in n_classesxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
