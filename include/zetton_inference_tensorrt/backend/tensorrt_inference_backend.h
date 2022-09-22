#pragma once

#include "zetton_inference/interface/base_inference_backend.h"

namespace zetton {
namespace inference {

class TensorRTInferenceBackend : public BaseInferenceBackend {
 public:
  int NumInputs() const final;
  int NumOutputs() const final;
  TensorInfo GetInputInfo(int index) final;
  TensorInfo GetOutputInfo(int index) final;
  bool Infer(std::vector<Tensor>& inputs, std::vector<Tensor>* outputs) final;
};

}  // namespace inference
}  // namespace zetton
