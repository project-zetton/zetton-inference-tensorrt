#include "zetton_inference_tensorrt/backend/tensorrt_inference_backend.h"

namespace zetton {
namespace inference {

int TensorRTInferenceBackend::NumInputs() const { return -1; }

int TensorRTInferenceBackend::NumOutputs() const { return -1; }

TensorInfo TensorRTInferenceBackend::GetInputInfo(int index) { return {}; }

TensorInfo TensorRTInferenceBackend::GetOutputInfo(int index) { return {}; }

bool TensorRTInferenceBackend::Infer(std::vector<Tensor>& inputs,
                                     std::vector<Tensor>* outputs) {
  return false;
}

ZETTON_REGISTER_INFERENCE_BACKEND(TensorRTInferenceBackend)

}  // namespace inference
}  // namespace zetton
