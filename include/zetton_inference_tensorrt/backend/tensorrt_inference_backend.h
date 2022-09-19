#pragma once

#include "zetton_inference/interface/base_inference_backend.h"

namespace zetton {
namespace inference {

struct TensorRTInferenceBackendInitOptions {};

class TensorRTInferenceBackend : public BaseInferenceBackend {
 public:
  TensorRTInferenceBackend(
      const TensorRTInferenceBackendInitOptions& init_options =
          TensorRTInferenceBackendInitOptions());

 public:
  bool Init(const std::map<std::string, std::vector<int>>& shapes) final;
  void Infer() final;
};

}  // namespace inference
}  // namespace zetton
