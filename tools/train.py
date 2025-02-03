import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fpseg import FpsegPredictor, sam_model_registry
from fpseg.datasets import CityscapesClasswise
from fpseg.utils import TrainLogger

import argparse
import os
import torchmetrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune prompts for FPSeg model based on pretrained image encoder and mask decoder."
    )

    parser.add_argument(
        "--config", default="configs/train_config.yaml", help="Train config file path."
    )

    args = parser.parse_args()

    return args


def train_loop(predictor, train_dataloader, optimizer, loss_fn, iou, config, train_logger):
    predictor.model.prompt_encoder.train()
    if config["tuned_part"] == "prompts and decoder":
        predictor.model.mask_decoder.train()

    for batch, (image, image_embedding, target, class_id) in tqdm(
        enumerate(train_dataloader), desc="Train batches", position=1, leave=False
    ):
        target = target.cuda()
        predictor.set_torch_embeddings(image_embedding.cuda(), target.shape[-2:])

        masks = predictor.predict(
            class_id=class_id,
            return_logits=True,
        )

        loss = loss_fn(masks.squeeze(), target.float().squeeze())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not batch % config["train_log_interval"]:
            loss_value = loss.item()
            iou_value = iou(masks.squeeze(), target.squeeze())
            iou.reset()

            values_str = f"loss: {loss_value:>7f}, mIoU: {iou_value:>4f}"
            tqdm.write(values_str)

            current = batch * config["batch_size"] + len(class_id)
            size = len(train_dataloader.dataset)

            train_logger.append(values_str + f" [{current:>5d}/{size:>5d}]")


def test_loop(predictor, test_dataloader, loss_fn, iou, config, train_logger, best_iou):
    predictor.model.prompt_encoder.eval()
    if config["tuned_part"] == "prompts and decoder":
        predictor.model.mask_decoder.eval()

    iou.reset()
    num_batches = len(test_dataloader)
    test_loss = 0

    with torch.no_grad():
        for batch, (image, image_embedding, target, class_id) in tqdm(
            enumerate(test_dataloader), desc="Test batches", position=1
        ):
            target = target.cuda()
            predictor.set_torch_embeddings(image_embedding.cuda(), target.shape[-2:])

            masks = predictor.predict(
                class_id=class_id,
                return_logits=True,
            )

            test_loss += loss_fn(masks.squeeze(), target.float().squeeze()).item()

            iou.update(masks.squeeze(), target.squeeze())

        loss_value = test_loss / num_batches
        test_iou = iou.compute()
        iou.reset()

        values_str = f"loss: {loss_value:>7f}, mIoU: {test_iou:>4f}"
        tqdm.write("Test results: " + values_str)

        train_logger.append(f"\n Test_result: {values_str} \n")

        if test_iou > best_iou:
            best_iou = test_iou

            if config["save_checkpoint"]:
                torch.save(
                    predictor.model.state_dict(), os.path.join(train_logger.get_dir(), "checkpoint.pth")
                )
        return best_iou


def train(config):
    # Data
    train_dataset = CityscapesClasswise(
        config["data_root"],
        image_embedding_dir=config["image_embedding_dir"],
        split="train",
        mode="fine",
        p_neg_max=config["p_neg_max"],
        target_size=config["target_size"],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )

    test_dataset = CityscapesClasswise(
        config["data_root"],
        image_embedding_dir=config["image_embedding_dir"],
        split="val",
        mode="fine",
        n_class_examples_per_epoch=config["test_n_class_examples_per_epoch"],
        p_neg_max=config["p_neg_max"],
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"]
    )

    # Model
    fpseg = sam_model_registry[config["encoder"]](
        n_classes=config["n_classes"],
        prompt_size=config["prompt_size"],
        checkpoint=config["checkpoint"],
    ).cuda()

    predictor = FpsegPredictor(fpseg)

    # Train setup
    match config["tuned_part"]:
        case "prompts":
            optimizer = torch.optim.AdamW(
                fpseg.prompt_encoder.parameters(), lr=config["prompt_learning_rate"]
            )
        case "prompts and decoder":
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": fpseg.prompt_encoder.parameters(),
                        "lr": config["prompt_learning_rate"],
                    },
                    {
                        "params": fpseg.mask_decoder.parameters(),
                        "lr": config["decoder_learning_rate"],
                    },
                ]
            )

    loss_fn = torch.nn.BCEWithLogitsLoss()
    iou = torchmetrics.JaccardIndex(task="binary").cuda()
    best_iou = 0  # Used for saving best model weights

    # Training
    train_logger = TrainLogger(base_dir=config["log_dir"], config=config)
    for i in tqdm(range(config["n_epochs"]), desc="Epochs", position=0):
        train_logger.append(f"Epoch {i+1}:")
        train_loop(predictor, train_dataloader, optimizer, loss_fn, iou, config, train_logger)

        if i > 0 and not i % config["test_interval"]:
            best_iou = test_loop(predictor, test_dataloader, loss_fn, iou, config, train_logger, best_iou)

    final_message = f"Best test mIoU (model saved in checkpoint): {best_iou}"
    print(final_message)
    train_logger.append(f"\n {final_message} \n")


def main():
    args = parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    train(config)


if __name__ == "__main__":
    main()
