# zetton-inference-tensorrt

[English](README.md) | 中文

## 目录

- [zetton-inference-tensorrt](#zetton-inference-tensorrt)
  - [目录](#目录)
  - [简介](#简介)
  - [最新进展](#最新进展)
  - [安装](#安装)
  - [教程](#教程)
  - [基准测试和模型库](#基准测试和模型库)
  - [常见问题](#常见问题)
  - [贡献指南](#贡献指南)
  - [致谢](#致谢)
  - [许可证](#许可证)
  - [相关项目](#相关项目)

## 简介

zetton-inference-tensorrt 是 [zetton-inference](https://github.com/project-zetton/zetton-inference) 软件包的一个开源扩展，它使用 TensorRT 框架实现深度学习推理。它是 [Project Zetton](https://github.com/project-zetton) 的一部分。

## 最新进展

有关详细信息和发布历史，请参阅 [changelog.md](docs/zh_CN/changelog.md) 。

有关 zetton-inference-tensorrt 不同版本之间的兼容性更改，请参阅 [compatibility.md](docs/zh_CN/compatibility.md) 。

## 安装

请参考 [快速入门指南](docs/zh_CN/get_started.md) 获取安装说明。

## 教程

请查看 [快速入门指南](docs/zh_CN/get_started.md) 以了解 zetton-inference-tensorrt 的基本用法。

## 基准测试和模型库

NVIDIA Jetson Xavier NX

| 任务      | 模型      | FP32 | FP16 | INT8 |
| :-------- | :-------- | :--- | :--- | :--- |
| Detection | YOLOv5    |      |      |      |
| Detection | YOLOX     |      |      |      |
| Detection | YOLOv7    |      |      |      |
| Tracking  | DeepSORT  |      |      |      |
| Tracking  | ByteTrack |      |      |      |

装载有 NVIDIA A6000 GPU 的 x86 服务器

| 任务      | 模型      | FP32 | FP16 | INT8 |
| :-------- | :-------- | :--- | :--- | :--- |
| Detection | YOLOv5    |      |      |      |
| Detection | YOLOX     |      |      |      |
| Detection | YOLOv7    |      |      |      |
| Tracking  | DeepSORT  |      |      |      |
| Tracking  | ByteTrack |      |      |      |

## 常见问题

请参考 [FAQ](docs/zh_CN/faq.md) 获取常见问题的答案。

## 贡献指南

我们感谢所有为改进 zetton-inference-tensorrt 做出贡献的人。请参考 [贡献指南](.github/CONTRIBUTING.md) 了解参与项目贡献的相关指引。

## 致谢

我们感谢所有实现自己方法或添加新功能的贡献者，以及给出有价值反馈的用户。

我们希望这个软件包和基准能够通过提供灵活的工具包来部署模型，为不断发展的研究和生产社区服务。

## 许可证

- 对于学术用途，该项目在 2 条款 BSD 许可证下授权，请参阅 [LICENSE 文件](LICENSE) 以获取详细信息。

- 对于商业用途，请联系 [Yusu Pan](mailto:xxdsox@gmail.com)。

## 相关项目

- [zetton-inference](https://github.com/project-zetton/zetton-inference): 适用于深度学习推理的主要软件包

- [zetton-ros-vendor](https://github.com/project-zetton/zetton-ros-vendor): ROS 相关示例
