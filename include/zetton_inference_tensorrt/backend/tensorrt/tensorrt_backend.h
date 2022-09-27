#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "zetton_inference/base/options.h"
#include "zetton_inference/interface/base_inference_backend.h"
#include "zetton_inference_tensorrt/backend/tensorrt/buffers.h"
#include "zetton_inference_tensorrt/backend/tensorrt/common.h"
#include "zetton_inference_tensorrt/backend/tensorrt/util.h"

namespace zetton {
namespace inference {

struct TensorRTValueInfo {
  std::string name;
  std::vector<int> shape;
  nvinfer1::DataType dtype;
};

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

class TensorRTInferenceBackend : public BaseInferenceBackend {
 public:
  TensorRTInferenceBackend() : engine_(nullptr), context_(nullptr) {}
  ~TensorRTInferenceBackend() override {
    if (parser_) {
      parser_.reset();
    }
  }

 public:
  bool Init(const InferenceRuntimeOptions& options) override;
  bool InitFromSerialized(const TensorRTInferenceBackendOptions& options =
                              TensorRTInferenceBackendOptions());
  bool InitFromONNX(const std::string& model_file,
                    const TensorRTInferenceBackendOptions& options =
                        TensorRTInferenceBackendOptions());
  bool Infer(std::vector<Tensor>& inputs,
             std::vector<Tensor>* outputs) override;

  int NumInputs() const override { return inputs_desc_.size(); }
  int NumOutputs() const override { return outputs_desc_.size(); }
  TensorInfo GetInputInfo(int index) override;
  TensorInfo GetOutputInfo(int index) override;

 private:
  void GetInputOutputInfo();
  void SetInputs(const std::vector<Tensor>& inputs);
  void AllocateOutputsBuffer(std::vector<Tensor>* outputs);

 private:
  bool CreateTensorRTEngineFromSerialized();
  bool CreateTensorRTEngineFromONNX(const std::string& model_file);

  bool LoadTensorRTEngineFromSerialized(const std::string& trt_engine_file);
  bool BuildTensorRTEngineFromFromONNX();

 private:
  TensorRTInferenceBackendOptions options_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  tensorrt::UniquePtr<nvonnxparser::IParser> parser_;
  tensorrt::UniquePtr<nvinfer1::IBuilder> builder_;
  tensorrt::UniquePtr<nvinfer1::INetworkDefinition> network_;
  cudaStream_t stream_{};
  std::vector<void*> bindings_;
  std::vector<TensorRTValueInfo> inputs_desc_;
  std::vector<TensorRTValueInfo> outputs_desc_;
  std::map<std::string, tensorrt::DeviceBuffer> inputs_buffer_;
  std::map<std::string, tensorrt::DeviceBuffer> outputs_buffer_;

  // Sometimes while the number of outputs > 1
  // the output order of tensorrt may not be same
  // with the original onnx model
  // So this parameter will record to origin outputs
  // order, to help recover the rigt order
  std::map<std::string, int> outputs_order_;
};

}  // namespace inference
}  // namespace zetton
