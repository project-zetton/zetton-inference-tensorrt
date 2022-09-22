#pragma once

#include <NvInfer.h>

#include <memory>
#include <numeric>

namespace zetton {
namespace inference {
namespace tensorrt {

inline uint32_t getElementSize(nvinfer1::DataType t) noexcept {
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return 1;
  }
  return 0;
}

inline int64_t volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    delete obj;
  }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, InferDeleter>;

}  // namespace tensorrt
}  // namespace inference
}  // namespace zetton
