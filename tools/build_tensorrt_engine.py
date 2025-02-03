import yaml

import argparse
import os

try:
    import tensorrt as trt
except ImportError:
    raise ImportError(
        "Install optional dependencies to build TensorRT engine with: pip install -e .[all]"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build TensorRT engine from ONNX model. Must be executed on the device the"
            "model is supposed to run on!"
        )
    )

    parser.add_argument(
        "--config",
        default="configs/build_tensorrt_engine_config.yaml",
        help="Build TensorRT engine config file path.",
    )

    args = parser.parse_args()

    return args


def build_tensorrt_engine(config):
    logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(logger)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    parser = trt.OnnxParser(network, logger)

    success = parser.parse_from_file(config["onnx_file"])
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        raise Exception("Parsing onnx did not work")

    trt_config = builder.create_builder_config()
    trt_filename = os.path.splitext(os.path.basename(config["onnx_file"]))[0] + ".trt"

    serialized_engine = builder.build_serialized_network(network, trt_config)

    with open(os.path.join(config["output_dir"], trt_filename), "wb") as f:
        f.write(serialized_engine)


def main():
    args = parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    if not os.path.isdir(config["output_dir"]):
        os.makedirs(config["output_dir"], exist_ok=True)

    build_tensorrt_engine(config)


if __name__ == "__main__":
    main()
