#pragma once

#include <NvInfer.h>
#include <absl/strings/str_join.h>
#include <fmt/format.h>
#include <zetton_inference/base/type.h>

#include <map>
#include <vector>

namespace zetton {
namespace inference {
namespace tensorrt {

struct ShapeRangeInfo {
 public:
  explicit ShapeRangeInfo(const std::vector<int64_t>& new_shape) {
    shape.assign(new_shape.begin(), new_shape.end());
    min.resize(new_shape.size());
    max.resize(new_shape.size());
    is_static.resize(new_shape.size());
    for (size_t i = 0; i < new_shape.size(); ++i) {
      if (new_shape[i] > 0) {
        min[i] = new_shape[i];
        max[i] = new_shape[i];
        is_static[i] = 1;
      } else {
        min[i] = -1;
        max[i] = -1;
        is_static[i] = 0;
      }
    }
  }

 public:
  /// \brief update with given shape
  /// \return shape validity
  /// -1: new shape is inillegal
  /// 0 : new shape is able to inference
  /// 1 : new shape is out of range, need to update engine
  int Update(const std::vector<int64_t>& new_shape);
  int Update(const std::vector<int>& new_shape) {
    std::vector<int64_t> new_shape_int64(new_shape.begin(), new_shape.end());
    return Update(new_shape_int64);
  }

 public:
  std::string name;
  std::vector<int64_t> shape;
  std::vector<int64_t> min;
  std::vector<int64_t> max;
  std::vector<int64_t> opt;
  std::vector<int8_t> is_static;
};

std::string ToString(const ShapeRangeInfo& info);

/// \brief check whether or not the model can build tensorrt engine now.
/// \details if the model has dynamic input shape, it will require defined shape
/// information We can set the shape range information by function
/// SetTrtInputShape(). But if the shape range is not defined, then the engine
/// cannot build, in this case, the engine will build once there's data feeded,
/// and the shape range will be updated
bool CanBuildEngine(
    const std::map<std::string, ShapeRangeInfo>& shape_range_info);

/// \brief get corresponding inference data type from tensorrt data type
InferenceDataType GetInferenceDataType(const nvinfer1::DataType& dtype);

/// \brief convert int vector to nvinfer1::Dims
nvinfer1::Dims ToDims(const std::vector<int>& vec);
/// \brief convert int64_t vector to nvinfer1::Dims
nvinfer1::Dims ToDims(const std::vector<int64_t>& vec);
/// \brief convert nvinfer1::Dims to int vector
std::vector<int> ToVec(const nvinfer1::Dims& dim);

}  // namespace tensorrt
}  // namespace inference
}  // namespace zetton
