#include "zetton_inference_tensorrt/backend/tensorrt/util.h"

#include "zetton_common/log/log.h"

namespace zetton {
namespace inference {
namespace tensorrt {

std::string ToString(const ShapeRangeInfo& info) {
  return fmt::format("Input nmae: {}, shape: {}, min={}, max={}", info.name,
                     absl::StrJoin(info.shape, " "),
                     absl::StrJoin(info.min, " "),
                     absl::StrJoin(info.max, " "));
}

bool CanBuildEngine(
    const std::map<std::string, ShapeRangeInfo>& shape_range_info) {
  for (const auto& info : shape_range_info) {
    bool is_full_static = true;
    for (long dim : info.second.shape) {
      if (dim < 0) {
        is_full_static = false;
        break;
      }
    }

    if (is_full_static) {
      continue;
    }
    for (size_t i = 0; i < info.second.shape.size(); ++i) {
      if (info.second.min[i] < 0 || info.second.max[i] < 0) {
        return false;
      }
    }
  }
  return true;
}

InferenceDataType GetInferenceDataType(const nvinfer1::DataType& dtype) {
  if (dtype == nvinfer1::DataType::kFLOAT) {
    return InferenceDataType::kFP32;
  } else if (dtype == nvinfer1::DataType::kHALF) {
    return InferenceDataType::kFP16;
  } else if (dtype == nvinfer1::DataType::kINT8) {
    return InferenceDataType::kINT8;
  } else if (dtype == nvinfer1::DataType::kINT32) {
    return InferenceDataType::kINT32;
  }
  return InferenceDataType::kBOOL;
}

nvinfer1::Dims ToDims(const std::vector<int>& vec) {
  int limit = static_cast<int>(nvinfer1::Dims::MAX_DIMS);
  if (static_cast<int>(vec.size()) > limit) {
    AWARN_F("Vector too long, only first 8 elements are used in dimension.");
  }
  // Pick first nvinfer1::Dims::MAX_DIMS elements
  nvinfer1::Dims dims{std::min(static_cast<int>(vec.size()), limit), {}};
  std::copy_n(vec.begin(), dims.nbDims, std::begin(dims.d));
  return dims;
}

nvinfer1::Dims ToDims(const std::vector<int64_t>& vec) {
  int limit = static_cast<int>(nvinfer1::Dims::MAX_DIMS);
  if (static_cast<int>(vec.size()) > limit) {
    AWARN_F("Vector too long, only first 8 elements are used in dimension.");
  }
  // Pick first nvinfer1::Dims::MAX_DIMS elements
  nvinfer1::Dims dims{std::min(static_cast<int>(vec.size()), limit), {}};
  std::copy_n(vec.begin(), dims.nbDims, std::begin(dims.d));
  return dims;
}

std::vector<int> ToVec(const nvinfer1::Dims& dim) {
  std::vector<int> out(dim.d, dim.d + dim.nbDims);
  return out;
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace zetton
