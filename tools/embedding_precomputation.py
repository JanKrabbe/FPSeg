import numpy as np
import yaml
import torch
from torchvision.transforms.functional import resize
from tqdm import tqdm

from fpseg import FpsegPredictor, sam_model_registry

import argparse
import os
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Precomputed SAM image embeddings")

    parser.add_argument(
        "--config",
        default="configs/embedding_precomputation.yaml",
        help="Embedding precomputation config file path.",
    )

    args = parser.parse_args()

    return args


def get_image_files(root: str, split: str) -> list:
    """
    Creates a list of all image files.

    Arguments:
        root (str): Directory of the Cityscapes dataset.
        split (str): Split of the dataset.
    """
    images_dir = os.path.join(root, "leftImg8bit", split)
    image_files = []

    if not os.path.isdir(images_dir):
        print(
            f"Dataset split {split} not found. No embeddings are calculated for this split. "
            f"Please make sure the root directory exists and contains the {split} folder."
        )
        return image_files

    for city in os.listdir(images_dir):
        img_dir = os.path.join(images_dir, city)
        for file_name in os.listdir(img_dir):
            image_files.append(os.path.join(img_dir, file_name))

    return image_files


def calculate_embeddings(
    image_files: list[str],
    predictor: FpsegPredictor,
    embedding_dir: str,
    image_size: tuple[int, int] = None,
) -> None:
    """
    Calculates and saves embeddings for all image files.

    Arguments:
       images_files (list[str]): List of image file names for which embeddings are to be calculated.
       predictor (FpsegPredictor): Predictor used to calculate the embeddings. Only the image
           encoder is relevant.
       embedding_dir (str): Embeddings are saved in this directory.
       image_size (tuple[int, int], optional): If a size in (HxW) format is given, the image is
           resized before calculating embeddings.
    """
    for image_file in tqdm(image_files):
        image_id = image_file.split("/")[-1][:-4]
        image = Image.open(image_file).convert("RGB")

        if image_size != (0, 0):
            image = resize(image, image_size)

        predictor.set_image(np.array(image))
        image_embedding = predictor.features[0]

        torch.save(image_embedding, os.path.join(embedding_dir, image_id + ".pt"))


def main():
    args = parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    sam = sam_model_registry[config["encoder"]](
        n_classes=1, prompt_size=1, checkpoint=config["checkpoint"]
    ).cuda()  # n_classes and prompt_size doesn't matter in this case
    
    predictor = FpsegPredictor(sam)

    for split in config["splits"]:
        images_files = get_image_files(config["data_root"], split)
        embedding_dir = os.path.join(config["data_root"], config["image_embedding_dir"], split)

        if len(images_files) > 0:
            print(f"Calculating embeddings for {split} split:")

            if not os.path.exists(embedding_dir):
                os.makedirs(embedding_dir)

            image_size = config["image_size"]
            calculate_embeddings(images_files, predictor, embedding_dir, image_size)


if __name__ == "__main__":
    main()
