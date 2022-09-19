#include "zetton_inference_tensorrt/backend/tensorrt_inference_backend.h"

namespace zetton {
namespace inference {

TensorRTInferenceBackend::TensorRTInferenceBackend(
    const TensorRTInferenceBackendInitOptions& init_options) {}

bool TensorRTInferenceBackend::Init(
    const std::map<std::string, std::vector<int>>& shapes) {
  return false;
}

void TensorRTInferenceBackend::Infer() {}

ZETTON_REGISTER_INFERENCE_BACKEND(TensorRTInferenceBackend)

}  // namespace inference
}  // namespace zetton
