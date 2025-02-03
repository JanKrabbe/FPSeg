# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="fpseg",
    version="1.0",
    install_requires=[
        "tqdm>=4.65",
        "torchmetrics>=1.1.1",
        "timm"
    ],
    packages=find_packages(),
    extras_require={
        "all": ["tensorrt==8.6.1", "pycuda>=2022.1", "onnx"],
        "dev": ["flake8", "isort==5.12.0", "black~=23.1", "mypy"],
    },
)
