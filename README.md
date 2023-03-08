# zetton-inference-tensorrt

English | [中文](README_zh-CN.md)

## Table of Contents

- [zetton-inference-tensorrt](#zetton-inference-tensorrt)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [What's New](#whats-new)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
  - [Overview of Benchmark and Model Zoo](#overview-of-benchmark-and-model-zoo)
  - [FAQ](#faq)
  - [Contributing](#contributing)
  - [Acknowledgement](#acknowledgement)
  - [License](#license)
  - [Related Projects](#related-projects)

## Introduction

zetton-inference-tensorrt is an open source extension of [zetton-inference](https://github.com/project-zetton/zetton-inference) package that enables deep learning inference using the TensorRT framework. It's a part of the [Project Zetton](https://github.com/project-zetton).

## What's New

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

For compatibility changes between different versions of zetton-inference, please refer to [compatibility.md](docs/en/compatibility.md).

## Installation

Please refer to [Installation](docs/en/get_started.md) for installation instructions.

## Getting Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of zetton-inference.

## Overview of Benchmark and Model Zoo

NVIDIA Jetson Xavier NX

| Task      | Model     | FP32 | FP16 | INT8 |
| :-------- | :-------- | :--- | :--- | :--- |
| Detection | YOLOv5    |      |      |      |
| Detection | YOLOX     |      |      |      |
| Detection | YOLOv7    |      |      |      |
| Tracking  | DeepSORT  |      |      |      |
| Tracking  | ByteTrack |      |      |      |

x86 with NVIDIA A6000 GPU

| Task      | Model     | FP32 | FP16 | INT8 |
| :-------- | :-------- | :--- | :--- | :--- |
| Detection | YOLOv5    |      |      |      |
| Detection | YOLOX     |      |      |      |
| Detection | YOLOv7    |      |      |      |
| Tracking  | DeepSORT  |      |      |      |
| Tracking  | ByteTrack |      |      |      |

## FAQ

Please refer to [FAQ](docs/en/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve zetton-inferenece. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the package and benchmark could serve the growing research and production community by providing a flexible toolkit to deploy models.

## License

- For academic use, this project is licensed under the 2-clause BSD License, please see the [LICENSE file](LICENSE) for details.

- For commercial use, please contact [Yusu Pan](mailto:xxdsox@gmail.com).

## Related Projects

- [zetton-inference](https://github.com/project-zetton/zetton-inference): main package for deep learning inference.

- [zetton-ros-vendor](https://github.com/project-zetton/zetton-ros-vendor):
ROS-related examples.
