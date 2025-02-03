import numpy as np
import torch
from torchvision.datasets.utils import extract_archive, iterable_to_str, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms.functional import InterpolationMode, resize
from tqdm import tqdm

import os
import pickle
import random
from collections import namedtuple
from PIL import Image
from typing import Callable, Optional, Tuple


class CityscapesClasswise(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset. The dataset is used for prompt
      tuning. Each sample consists of an image, a precomputed image embedding, a binary segmentation
      mask of a single class, and its corresponding ID.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        image_embedding_dir (string): Directory of the precomputed image_embeddings inside root
        (computed with tools/embedding_precomputation.py)
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if
            mode="fine" otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine``, ``coarse`` or ``refined``
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target
            as entry and returns a transformed version.
        n_class_examples_per_epoch (int, optional): Defines the number of times each class is used
            in one epoch.
        p_neg_max (float, optional): Defines the maximum usage percentage of negative examples for
            all classes. Important for the prompt tuning of rare classes.
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
        n_class_examples_per_epoch: int = 100,
        p_neg_max: float = 0.2,
        target_size: Tuple[int, int] = (1024, 1024),
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.mode = "gtFine" if mode == "fine" else "gtCoarse"
        self.images_dir = os.path.join(self.root, "leftImg8bit", split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.split = split
        self.images = []
        self.targets = []
        self.n_class_examples_per_epoch = n_class_examples_per_epoch
        self.p_neg_max = p_neg_max
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

        if p_neg_max < 0.0 or p_neg_max > 1.0:
            raise ValueError(f"p_neg_max should be in [0,1] but is {p_neg_max}")

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            if split == "train_extra":
                image_dir_zip = os.path.join(self.root, "leftImg8bit_trainextra.zip")
            else:
                image_dir_zip = os.path.join(self.root, "leftImg8bit_trainvaltest.zip")

            if self.mode == "gtFine":
                target_dir_zip = os.path.join(self.root, f"{self.mode}_trainvaltest.zip")
            elif self.mode == "gtCoarse":
                target_dir_zip = os.path.join(self.root, f"{self.mode}.zip")

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
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
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_name = "{}_{}".format(
                    file_name.split("_leftImg8bit")[0], f"{self.mode}_labelIds.png"
                )

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(os.path.join(target_dir, target_name))

        # Prepare datastructures that hold class names and IDs. Categories are considered as classes
        # with a list of IDs.
        self.used_classes = {}
        self.used_class_names = []
        for cityscapes_class in self.classes[7:-1]:
            self.used_classes[cityscapes_class.name] = [cityscapes_class.id]
            self.used_class_names.append(cityscapes_class.name)
            if cityscapes_class.category not in self.used_classes:
                self.used_classes[cityscapes_class.category] = [cityscapes_class.id]
                self.used_class_names.append(cityscapes_class.category)
            else:
                self.used_classes[cityscapes_class.category].append(cityscapes_class.id)

        # Load or prepare dicts that are used to sample positive and negative images. For a given
        # class a negative image doesn't contain the class while a positive image contians the class
        if os.path.isfile(
            os.path.join(root, "leftImg8bit", "pos_classwise_class_dict_" + split + ".pkl")
        ):
            with open(
                os.path.join(root, "leftImg8bit", "pos_classwise_class_dict_" + split + ".pkl"),
                "rb",
            ) as f:
                self.pos_class_dict = pickle.load(f)
            with open(
                os.path.join(root, "leftImg8bit", "neg_classwise_class_dict_" + split + ".pkl"),
                "rb",
            ) as f:
                self.neg_class_dict = pickle.load(f)
        else:
            print("Saved classwise class dicts not found. Calculating:")
            self.pos_class_dict = {k: [] for k in self.used_classes.keys()}
            self.neg_class_dict = {k: [] for k in self.used_classes.keys()}
            for target_id, target_path in tqdm(enumerate(self.targets)):
                target = Image.open(target_path[0])
                target = np.array(target)
                target = np.unique(target)

                for k in self.used_classes.keys():
                    if len(np.intersect1d(self.used_classes[k], target)) > 0:
                        self.pos_class_dict[k].append(target_id)
                    else:
                        self.neg_class_dict[k].append(target_id)

            with open(
                os.path.join(root, "leftImg8bit", "pos_classwise_class_dict_" + split + ".pkl"),
                "wb",
            ) as f:
                pickle.dump(self.pos_class_dict, f)
            with open(
                os.path.join(root, "leftImg8bit", "neg_classwise_class_dict_" + split + ".pkl"),
                "wb",
            ) as f:
                pickle.dump(self.neg_class_dict, f)

    def __len__(self) -> int:
        return len(self.used_classes) * self.n_class_examples_per_epoch

    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor, int]:
        """
        Args:
            index (int): Index
        Returns:
            (torch.tensor): The sampled image in 3xHxW format.
            (torch.tensor): Precomputed embedding of the sampled image in 256x64x64 format.
            (torch.tensor): Binary mask of the target class in the sampled image in HxW format.
            (int): ID of the targed class.
        """
        class_id = index % len(self.used_classes)

        class_name = self.used_class_names[class_id]
        n_pos = len(self.pos_class_dict[class_name])
        n_neg = len(self.neg_class_dict[class_name])

        # No more than 1-self.p_neg_max % neg examples per class are used
        p_pos = max(n_pos / (n_pos + n_neg), (1 - self.p_neg_max))

        # Sample random pos oder neg image
        if torch.rand(1) < p_pos and len(self.pos_class_dict[class_name]) > 0:
            image_index = random.choice(self.pos_class_dict[class_name])
        else:
            image_index = random.choice(self.neg_class_dict[class_name])

        return *self._load_single_item(image_index, class_name), class_id

    def _load_single_item(
        self, index: int, class_name: str
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Args:
            index (int): Index
            class_name (string): Target class of the current sample. Used to create
              the binary mask.

        Returns:
            (torch.tensor): The sampled image in 3xHxW format.
            (torch.tensor): Precomputed embedding of the sampled image in 256x64x64 format.
            (torch.tensor): Binary mask of the target class in the sampled image in HxW format.
        """
        image = Image.open(self.images[index]).convert("RGB")
        image = torch.tensor(np.moveaxis(np.array(image), 2, 0))

        target = torch.tensor(np.array(Image.open(self.targets[index])))

        if "train" in self.split and target.shape != self.target_size:
            image = resize(image, self.target_size, antialias=True)
            target = resize(
                target.unsqueeze(0),
                self.target_size,
                interpolation=InterpolationMode.NEAREST,
                antialias=True,
            ).squeeze()

        image_id = self.images[index].split("/")[-1][:-4]
        image_embedding = torch.load(
            os.path.join(self.root, self.image_embedding_dir, self.split, image_id + ".pt"),
            map_location=torch.device("cpu"),
        )

        # Handle categories (with multiple IDs) while creating binary target
        binary_target = None
        for c in self.used_classes[class_name]:
            if binary_target is None:
                binary_target = target == c
            else:
                binary_target = torch.logical_or(binary_target, (target == c))

        return image, image_embedding, binary_target
