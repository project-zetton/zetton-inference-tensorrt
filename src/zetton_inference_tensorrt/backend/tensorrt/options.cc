#include "zetton_inference_tensorrt/backend/tensorrt/options.h"

namespace zetton {
namespace inference {
namespace tensorrt {

void TensorRTInferenceRuntimeOptions::SetInputShapeForTensorRT(
    const std::string& input_name, const std::vector<int32_t>& min_shape,
    const std::vector<int32_t>& opt_shape,
    const std::vector<int32_t>& max_shape) {
  trt_min_shape[input_name].clear();
  trt_max_shape[input_name].clear();
  trt_opt_shape[input_name].clear();
  trt_min_shape[input_name].assign(min_shape.begin(), min_shape.end());
  if (opt_shape.size() == 0) {
    trt_opt_shape[input_name].assign(min_shape.begin(), min_shape.end());
  } else {
    trt_opt_shape[input_name].assign(opt_shape.begin(), opt_shape.end());
  }
  if (max_shape.size() == 0) {
    trt_max_shape[input_name].assign(min_shape.begin(), min_shape.end());
  } else {
    trt_max_shape[input_name].assign(max_shape.begin(), max_shape.end());
  }
}

void TensorRTInferenceRuntimeOptions::EnableFP16ForTensorRT() {
  trt_enable_fp16 = true;
}

void TensorRTInferenceRuntimeOptions::DisableFP16ForTensorRT() {
  trt_enable_fp16 = false;
}

void TensorRTInferenceRuntimeOptions::SetCacheFileForTensorRT(
    const std::string& cache_file_path) {
  trt_serialize_file = cache_file_path;
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace zetton
