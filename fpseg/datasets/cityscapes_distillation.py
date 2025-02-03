import numpy as np
import torch
from torchvision.datasets.utils import extract_archive, iterable_to_str, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms.functional import resize

import os
from collections import namedtuple
from PIL import Image
from typing import Callable, Optional, Tuple


class CityscapesDistillation(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset. The dataset is used for
      distilling the MobileSAM image encoder. Each sample consists of an image and precomputed image
      embedding obtained with the SAM image encoder.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        image_embedding_dir (string): Directory of the precomputed image_embeddings (computed with
            tools/embedding_precomputation.py)
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if
            mode="fine" otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine``, ``coarse`` or ``refined``
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target
            as entry and returns a transformed version.
        target_size (tuple(int,int), optional): The size of the sampled image and binary mask in
            (H,W) format.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple(
        "CityscapesClass",
        [
            "name",
            "id",
            "train_id",
            "category",
            "category_id",
            "has_instances",
            "ignore_in_eval",
            "color",
        ],
    )

    classes = [
        CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
        CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
        CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
        CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
        CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
        CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
        CityscapesClass("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
        CityscapesClass("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
        CityscapesClass("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
        CityscapesClass("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
        CityscapesClass("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
        CityscapesClass("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
        CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
        CityscapesClass("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
        CityscapesClass("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
        CityscapesClass("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
        CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
        CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
        CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
        CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
        CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
        CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
        CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
        CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
        CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
        CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
        CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        CityscapesClass("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
    ]

    def __init__(
        self,
        root: str,
        image_embedding_dir: str = "image_embeddings",
        split: str = "train",
        mode: str = "fine",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        target_size: Tuple[int, int] = (1024, 1024),
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.mode = "gtFine" if mode == "fine" else "gtCoarse"
        self.images_dir = os.path.join(self.root, "leftImg8bit", split)
        self.split = split
        self.images = []
        self.target_size = target_size
        self.image_embedding_dir = image_embedding_dir

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = "Unknown value '{}' for argument split if mode is '{}'. Valid values are {{{}}}."
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not os.path.isdir(self.images_dir):
            if split == "train_extra":
                image_dir_zip = os.path.join(self.root, "leftImg8bit_trainextra.zip")
            else:
                image_dir_zip = os.path.join(self.root, "leftImg8bit_trainvaltest.zip")

            if os.path.isfile(image_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
            else:
                raise RuntimeError(
                    "Dataset not found or incomplete. Please make sure all required folders for the"
                    ' specified "split" and "mode" are inside the "root" directory'
                )

        if not os.path.isdir(os.path.join(self.root, self.image_embedding_dir)):
            raise RuntimeError(
                "Precomputed image embeddings not found. Please make sure to use the correct path"
                "or run tools/embedding_precomputation.py to get the embeddings."
            )

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor, int]:
        """
        Args:
            index (int): Index
        Returns:
            (torch.tensor): The sampled image in 3xHxW format.
            (torch.tensor): Precomputed embedding of the sampled image in 256x64x64 format.
        """

        # Workaround needed for using multiple workers
        os.system("taskset -p 0xffffffffffffffff %d >/dev/null 2>&1" % os.getpid())
        # os.system("taskset -p 0xffffffffffffffff %d" % os.getpid())

        image = Image.open(self.images[index]).convert("RGB")
        image = torch.tensor(np.moveaxis(np.array(image), 2, 0))

        if image.shape != self.target_size:
            image = resize(image, self.target_size, antialias=True)

        image_id = self.images[index].split("/")[-1][:-4]
        image_embedding = torch.load(
            os.path.join(self.root, self.image_embedding_dir, self.split, image_id + ".pt"),
            map_location=torch.device("cpu"),
        )

        return image, image_embedding
