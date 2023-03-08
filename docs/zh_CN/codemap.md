---
title: 代码地图
---

- [代码地图](#代码地图)
  - [目录结构](#目录结构)
  - [模块介绍](#模块介绍)

# 代码地图

本文档旨在帮助您了解 zetton-common 代码的结构和功能。

## 目录结构

zetton-inference-tensorrt 代码的目录结构如下：

```bash
$ tree -L 3

.
├── cmake/
├── CMakeLists.txt
├── docs/
├── examples/
├── include/
├── LICENSE
├── package.xml
├── README.md
├── README_zh-CN.md
├── src/
├── tests/
└── tools/
```

其中：

- `.github/`：GitHub 相关的配置文件
- `cmake/` 与 `CMakeLists.txt`：CMake 构建相关的文件
- `docs/`：文档目录
- `examples/`：示例代码目录
- `include/`：头文件目录
- `src/`：源代码目录
- `tests/`：测试代码目录
- `tools/`：工具脚本目录
- `LICENSE`：软件包许可证
- `README.md`：软件包说明文档
- `README_zh-CN.md`：软件包说明文档（中文）
- `package.xml`：软件包描述文件，用于 colcon 构建

## 模块介绍

zetton-inference-tensorrt 代码包含如下模块：

- `backend`：后端模块，包含 TensorRT 后端的实现
- `util`：工具模块，包含一些常用的工具函数
- `vision`：视觉模块，包含一些视觉相关的模型和算法
