import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn

from fpseg import sam_model_registry

import argparse
import time

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt

    trt_packages_exist = True
except ImportError:
    trt_packages_exist = False


cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark FPSeg model to get inference times.")

    parser.add_argument(
        "--config", default="configs/benchmark_config.yaml", help="Benchmark config file path."
    )

    args = parser.parse_args()

    return args


def benchmark_torch(config, input_data):
    input_data = input_data.cuda()
    fpseg = sam_model_registry[config["encoder"]](
        n_classes=config["n_classes"], prompt_size=config["prompt_size"]
    ).cuda()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(config["n_warmup"]):
            masks = fpseg(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, config["n_runs"] + 1):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            masks = fpseg(input_data)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                avg_time_ms = np.mean(timings) * 1000
                print(f'Iteration {i}/{config["n_runs"]}, avg batch time {avg_time_ms:.2f} ms')
    avg_throughput = 1 / np.mean(timings)
    print(f"Average throughput: {avg_throughput:.2f} images/second")
    result = {"timing_mean": np.mean(timings) * 1000, "timing_std": np.std(timings) * 1000}

    return result


def benchmark_trt(config, input_data):
    input_data = input_data.numpy()

    f = open(config["trt_file"], "rb")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    output = np.empty([config["n_classes"], 1, 1024, 2048], dtype=np.bool_)

    # allocate device memory
    d_input = cuda.mem_alloc(1 * input_data.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    def predict(data):  # result gets copied into output
        # transfer input data to device
        cuda.memcpy_htod_async(d_input, data, stream)

        # execute model
        context.execute_async_v2(bindings, stream.handle, None)

        # syncronize threads
        stream.synchronize()

        return output

    print("Warm up ...")
    for _ in range(config["n_warmup"]):
        pred = predict(input_data)
    print("Start timing ...")
    timings = []
    for i in range(1, config["n_runs"] + 1):
        start_time = time.perf_counter()
        pred = predict(input_data)
        # Synchronize in predict function and no torch.cuda.synchronize() since torch is not used
        end_time = time.perf_counter()
        timings.append(end_time - start_time)
        if i % 10 == 0:
            print(
                "Iteration %d/%d, avg batch time %.2f ms"
                % (i, config["n_runs"], np.mean(timings) * 1000)
            )
    print("Average throughput: %.2f images/second" % (1 / np.mean(timings)))
    result = {"timing_mean": np.mean(timings) * 1000, "timing_std": np.std(timings) * 1000}

    return result


def main():
    args = parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    dummy_data = torch.rand((1, 3, 1024, 1024))

    if config["trt_file"] == "None":
        result = benchmark_torch(config, dummy_data)
    else:
        if not trt_packages_exist:
            raise ImportError(
                "Install optional dependencies to use TensorRT model with: pip install -e .[all]"
            )
        result = benchmark_trt(config, dummy_data)

    print(
        f"Mean inference duration {result['timing_mean']:>7f} ms with standard deviation "
        f"{result['timing_std']:>7f} ms"
    )


if __name__ == "__main__":
    main()
