# Code adapted for FPSeg

# Original file: segment-anything/predictor.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch

from typing import Tuple

from .modeling import FpsegSam
from .utils.transforms import ResizeLongestSide


class FpsegPredictor:
    def __init__(
        self,
        fpseg_model: FpsegSam,
    ) -> None:
        """
        Uses FPSegSAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          fpseg_model (FpsegSam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = fpseg_model
        self.transform = ResizeLongestSide(fpseg_model.image_encoder.img_size)
        self.reset_image()

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    @torch.no_grad()
    def set_torch_embeddings(
        self,
        embeddings: torch.Tensor,
        original_size=(1024, 1024),
        input_size=(1024, 1024),
    ) -> None:
        """
        Set pre calculated image embeddings, allowing masks to be
          predicted with the 'predict' method.

        Arguments:
          embeddings (torch.Tensor): The image embddings, with shape
            Bx256x64x64, where B is the batch size.
          original_image_size (tuple(int, int)): The size of the image
            used to calculate the embedding before transformation, in
            (H, W) format.
        """
        self.reset_image()

        self.original_size = original_size
        self.input_size = input_size
        self.features = embeddings
        self.is_image_set = True

    def predict(
        self,
        class_id: int,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for learned prompt embedding, using the currently set image or
        images if embeddings are set as batch.

        Arguments:
          class_id (int): The class to be segmented.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in BxHxW format, where B is the
            batch size and (H, W) is the original image size.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        masks = self.predict_torch(
            class_id,
            return_logits=return_logits,
        )

        return masks

    def predict_torch(
        self,
        class_id: int,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for learned prompt embedding, using the currently set image or
        images if embeddings are set as batch.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          class_id (int): The class to be segmented.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in Bx1xHxW format, where
            (H, W) is the original image size.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder()

        sparse_embeddings = sparse_embeddings[class_id, :, :]
        dense_embeddings = dense_embeddings.repeat(self.features.shape[0], 1, 1, 1)

        # Predict masks
        low_res_masks = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
