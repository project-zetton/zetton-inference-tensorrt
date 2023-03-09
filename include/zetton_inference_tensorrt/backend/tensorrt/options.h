#pragma once

#include "zetton_inference/base/options.h"

namespace zetton {
namespace inference {

/// \brief Options for TensorRT inference backend
struct TensorRTInferenceBackendOptions {
  /// \brief device id (e.g. GPU id) for model inference
  int gpu_id = 0;
  /// \brief whether or not to enable FP16 precision in TensorRT model inference
  bool enable_fp16 = false;
  /// \brief whether or not to enable INT8 precision in TensorRT model inference
  bool enable_int8 = false;
  /// \brief maximum batch size for TensorRT model inference
  size_t max_batch_size = 32;
  /// \brief maximum workspace size for TensorRT model inference
  size_t max_workspace_size = 1 << 30;
  /// \brief maximum input tensor shape for TensorRT model inference
  std::map<std::string, std::vector<int32_t>> max_shape;
  /// \brief minimum input tensor shape for TensorRT model inference
  std::map<std::string, std::vector<int32_t>> min_shape;
  /// \brief optimal input tensor shape for TensorRT model inference
  std::map<std::string, std::vector<int32_t>> opt_shape;
  /// \brief serialized TensorRT model file
  std::string serialize_file = "";
};

struct TensorRTInferenceRuntimeOptions : public InferenceRuntimeOptions {
 public:
  TensorRTInferenceRuntimeOptions() = default;
  ~TensorRTInferenceRuntimeOptions() override = default;

 public:
  /// \brief set input shape for TensorRT backend if the given model contains
  /// dynamic shape
  /// \details if opt_shape or max_shape are empty, they will keep same with the
  /// min_shape, which means the shape will be fixed as min_shape while
  /// inference
  /// \param input_name name of input tensor
  /// \param min_shape the minimum shape
  /// \param opt_shape the most common shape while inference, default be empty
  /// \param max_shape the maximum shape, default be empty
  void SetInputShapeForTensorRT(
      const std::string& input_name, const std::vector<int32_t>& min_shape,
      const std::vector<int32_t>& opt_shape = std::vector<int32_t>(),
      const std::vector<int32_t>& max_shape = std::vector<int32_t>());
  /// \brief enable half precision (FP16) for TensorRT backend
  void EnableFP16ForTensorRT();
  /// \brief disable half precision (FP16) and change to full precision (FP32)
  /// for TensorRT backend
  void DisableFP16ForTensorRT();
  /// \brief set path of cache file while using TensorRT backend
  /// \param cache_path path of cache file (serialized engine)
  void SetCacheFileForTensorRT(const std::string& cache_file_path);

 public:
  TensorRTInferenceBackendOptions backend_options;
};

}  // namespace inference
}  // namespace zetton
