import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fpseg import sam_model_registry
from fpseg.datasets import CityscapesDistillation
from fpseg.utils import TrainLogger

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill MobileSAM image encoder to cityscapes dataset."
    )

    parser.add_argument(
        "--config",
        default="configs/encoder_distillation_config.yaml",
        help="MobileSAM knowledge distillation config file path.",
    )

    args = parser.parse_args()

    return args


def train_loop(fpseg, train_dataloader, optimizer, loss_fn, config, train_logger):
    fpseg.image_encoder.train()

    for batch, (image, sam_image_embedding) in tqdm(
        enumerate(train_dataloader), desc="Train batches", position=1, leave=False
    ):
        input_image = fpseg.preprocess(image.cuda())
        embedding = fpseg.image_encoder(input_image)

        loss = loss_fn(embedding, sam_image_embedding.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not batch % config["train_log_interval"]:
            loss_value = loss.item()

            values_str = f"MSE: {loss_value:>7f}"
            tqdm.write(values_str)

            current = batch * config["batch_size"] + image.shape[0]
            size = len(train_dataloader.dataset)

            train_logger.append(values_str + f" [{current:>5d}/{size:>5d}]")


def test_loop(fpseg, test_dataloader, loss_fn, config, train_logger, best_test_loss):
    fpseg.image_encoder.eval()

    num_batches = len(test_dataloader)
    test_loss = 0

    with torch.no_grad():
        for batch, (image, sam_image_embedding) in tqdm(
            enumerate(test_dataloader), desc="Test batches", position=1
        ):
            input_image = fpseg.preprocess(image.cuda())
            embedding = fpseg.image_encoder(input_image)

            test_loss += loss_fn(embedding, sam_image_embedding.cuda()).item()

        loss_value = test_loss / num_batches

        values_str = f"MSE: {loss_value:>7f}"
        tqdm.write("Test results: " + values_str)

        train_logger.append(f"\n Test_result: {values_str} \n")

        if loss_value < best_test_loss:
            best_test_loss = loss_value

            if config["save_checkpoint"]: 
                torch.save(fpseg.state_dict(), os.path.join(train_logger.get_dir(), "checkpoint.pth"))
        return best_test_loss

def distill(config):
    # Data
    train_dataset = CityscapesDistillation(
        config["data_root"], split="train_extra", mode="coarse", target_size=config["target_size"]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    test_dataset = CityscapesDistillation(
        config["data_root"], split="val", mode="fine", target_size=config["target_size"]
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"]
    )

    # Model
    fpseg = sam_model_registry["mobilesam"](
        n_classes=config["n_classes"],
        prompt_size=config["prompt_size"],
        checkpoint=config["checkpoint"],
    ).cuda()

    # Train setup
    optimizer = torch.optim.AdamW(fpseg.image_encoder.parameters(), lr=config["learning_rate"])
    loss_fn = torch.nn.MSELoss()
    best_test_loss = float("inf")

    # Training
    train_logger = TrainLogger(base_dir=config["log_dir"], config=config)
    for i in tqdm(range(config["n_epochs"]), desc="Epochs", position=0):
        train_logger.append(f"Epoch {i+1}:")
        train_loop(fpseg, train_dataloader, optimizer, loss_fn, config, train_logger)

        if i > 0 and not i % config["test_interval"]:
            best_test_loss = test_loop(fpseg, test_dataloader, loss_fn, config, train_logger, best_test_loss)

    final_message = f"Best test MSE (model saved in checkpoint): {best_test_loss}"
    print(final_message)
    train_logger.append(f"\n {final_message} \n")


def main():
    args = parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    distill(config)


if __name__ == "__main__":
    main()
