import yaml
import torch

from fpseg import sam_model_registry
from fpseg.datasets import CityscapesDistillation

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Export trained FPSeg model to ONNX.")

    parser.add_argument(
        "--config", default="configs/onnx_export_config.yaml", help="ONNX export config file path."
    )

    args = parser.parse_args()

    return args


def onnx_export(config):
    dataset = CityscapesDistillation(
        config["data_root"], split="train", mode="fine", target_size=config["target_size"]
    )
    image, _ = dataset[0]

    fpseg = sam_model_registry[config["encoder"]](
        n_classes=config["n_classes"],
        prompt_size=config["prompt_size"],
        checkpoint=config["checkpoint"],
    ).cuda()

    torch.onnx.export(
        fpseg,
        image.unsqueeze(0).cuda(),
        os.path.join(
            config["output_dir"], f'fpseg_ps{config["prompt_size"]}_{config["encoder"]}.onnx'
        ),
        verbose=False,
    )


def main():
    args = parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    if not os.path.isdir(config["output_dir"]):
        os.makedirs(config["output_dir"], exist_ok=True)

    onnx_export(config)


if __name__ == "__main__":
    main()
