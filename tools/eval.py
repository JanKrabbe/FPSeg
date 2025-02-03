import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from tqdm import tqdm

from fpseg import FpsegPredictor, sam_model_registry
from fpseg.datasets import CityscapesClasswise

import argparse
import os
import torchmetrics

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt

    trt_packages_exist = True
except ImportError:
    trt_packages_exist = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained FPSeg checkpoint.")

    parser.add_argument(
        "--config", default="configs/eval_config.yaml", help="Eval config file path."
    )

    args = parser.parse_args()

    return args


def eval_torch(config, test_dataset):
    test_dataloader = DataLoader(
        test_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"]
    )

    # Model
    fpseg = sam_model_registry[config["encoder"]](
        n_classes=config["n_classes"],
        prompt_size=config["prompt_size"],
        checkpoint=config["checkpoint"],
    ).cuda()
    fpseg.eval()
    predictor = FpsegPredictor(fpseg)

    # Evaluation
    loss_fn = torch.nn.BCEWithLogitsLoss()
    iou = torchmetrics.JaccardIndex(task="binary").cuda()
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


            test_loss += loss_fn(masks.float().squeeze(), target.float().squeeze()).item()

            iou.update(masks.squeeze(), target.squeeze())

        loss_value = test_loss / len(test_dataloader)
        test_iou = iou.compute()

    return loss_value, test_iou


def eval_trt(config, test_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=config["num_workers"])
    example_input_batch, _, example_target, _ = test_dataset[0]

    f = open(config["checkpoint"], "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    output = np.empty(
        [config["n_classes"], 1, example_target.shape[-2], example_target.shape[-1]], dtype=np.bool_
    )

    # allocate device memory
    d_input = cuda.mem_alloc(1 * example_input_batch.unsqueeze(0).numpy().nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    iou = torchmetrics.JaccardIndex(task="binary").cuda()

    def predict(image_batch):  # result gets copied into output
        # transfer input data to device
        cuda.memcpy_htod_async(d_input, image_batch.numpy(), stream)
        # execute model
        context.execute_async_v2(bindings, stream.handle, None)
        # transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)

        # syncronize threads
        stream.synchronize()

        return output

    # Evaluation
    loss_fn = torch.nn.BCEWithLogitsLoss()
    iou = torchmetrics.JaccardIndex(task="binary").cuda()
    test_loss = 0

    with torch.no_grad():
        for batch, (image, image_embedding, target, class_id) in tqdm(
            enumerate(test_dataloader), desc="Test batches", position=1
        ):
            image = resize(image, (1024, 1024))
            target = target.cuda()

            masks = predict(image)

            masks = torch.tensor(masks[class_id]).squeeze().cuda()
            test_loss += loss_fn(masks.float(), target.float().squeeze()).item()

            iou.update(masks, target.squeeze())

        loss_value = test_loss / len(test_dataloader)
        test_iou = iou.compute()

    return loss_value, test_iou


def eval(config):
    # Data
    test_dataset = CityscapesClasswise(
        config["data_root"],
        image_embedding_dir=config["image_embedding_dir"],
        split="val",
        mode="fine",
        n_class_examples_per_epoch=config["test_n_class_examples_per_epoch"],
        p_neg_max=config["p_neg_max"],
    )

    if os.path.splitext(config["checkpoint"])[1] == ".trt":
        if not trt_packages_exist:
            raise ImportError(
                "Install optional dependencies to use TensorRT model with: pip install -e .[all]"
            )
        loss_value, test_iou = eval_trt(config, test_dataset)
    else:
        loss_value, test_iou = eval_torch(config, test_dataset)

    print(f"Eval results: loss: {loss_value:>7f}, mIoU: {test_iou:>4f}")


def main():
    args = parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    eval(config)


if __name__ == "__main__":
    main()
