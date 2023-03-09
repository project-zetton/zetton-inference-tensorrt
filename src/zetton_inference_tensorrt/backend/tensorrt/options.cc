#include "zetton_inference_tensorrt/backend/tensorrt/options.h"

namespace zetton {
namespace inference {

void TensorRTInferenceRuntimeOptions::SetInputShapeForTensorRT(
    const std::string& input_name, const std::vector<int32_t>& min_shape,
    const std::vector<int32_t>& opt_shape,
    const std::vector<int32_t>& max_shape) {
  backend_options.min_shape[input_name].clear();
  backend_options.max_shape[input_name].clear();
  backend_options.opt_shape[input_name].clear();
  backend_options.min_shape[input_name].assign(min_shape.begin(),
                                               min_shape.end());
  if (opt_shape.size() == 0) {
    backend_options.opt_shape[input_name].assign(min_shape.begin(),
                                                 min_shape.end());
  } else {
    backend_options.opt_shape[input_name].assign(opt_shape.begin(),
                                                 opt_shape.end());
  }
  if (max_shape.size() == 0) {
    backend_options.max_shape[input_name].assign(min_shape.begin(),
                                                 min_shape.end());
  } else {
    backend_options.max_shape[input_name].assign(max_shape.begin(),
                                                 max_shape.end());
  }
}

void TensorRTInferenceRuntimeOptions::EnableFP16ForTensorRT() {
  backend_options.enable_fp16 = true;
}

void TensorRTInferenceRuntimeOptions::DisableFP16ForTensorRT() {
  backend_options.enable_fp16 = false;
}

void TensorRTInferenceRuntimeOptions::SetCacheFileForTensorRT(
    const std::string& cache_file_path) {
  backend_options.serialize_file = cache_file_path;
}

}  // namespace inference
}  // namespace zetton
