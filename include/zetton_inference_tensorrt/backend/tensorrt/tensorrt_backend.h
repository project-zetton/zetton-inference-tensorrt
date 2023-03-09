#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "zetton_inference/base/options.h"
#include "zetton_inference/interface/base_inference_backend.h"
#include "zetton_inference_tensorrt/backend/tensorrt/buffers.h"
#include "zetton_inference_tensorrt/backend/tensorrt/common.h"
#include "zetton_inference_tensorrt/backend/tensorrt/options.h"
#include "zetton_inference_tensorrt/backend/tensorrt/util.h"

namespace zetton {
namespace inference {

/// \brief Input and output info of TensorRT inference engine
struct TensorRTValueInfo {
  std::string name;
  std::vector<int> shape;
  nvinfer1::DataType dtype;
};

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

/// \brief TensorRT inference backend
class TensorRTInferenceBackend : public BaseInferenceBackend {
 public:
  /// \brief constructor of TensorRTInferenceBackend
  TensorRTInferenceBackend() : engine_(nullptr), context_(nullptr) {}
  /// \brief destructor of TensorRTInferenceBackend
  ~TensorRTInferenceBackend() override {
    if (parser_) {
      parser_.reset();
    }
  }

 public:
  /// \brief initialize TensorRT inference backend with options
  bool Init(const InferenceRuntimeOptions* options) override;
  /// \brief initialize TensorRT inference backend with serialized model file
  bool InitFromSerialized(const TensorRTInferenceBackendOptions& options =
                              TensorRTInferenceBackendOptions());
  /// \brief initialize TensorRT inference backend with ONNX model file
  bool InitFromONNX(const std::string& model_file,
                    const TensorRTInferenceBackendOptions& options =
                        TensorRTInferenceBackendOptions());

  /// \brief infer the input tensors and save the results to output tensors
  bool Infer(std::vector<Tensor>& inputs,
             std::vector<Tensor>* outputs) override;

  /// \brief get the number of input tensors
  int NumInputs() const override { return inputs_desc_.size(); }
  /// \brief get the number of output tensors
  int NumOutputs() const override { return outputs_desc_.size(); }
  /// \brief get the input tensor info
  TensorInfo GetInputInfo(int index) override;
  /// \brief get the output tensor info
  TensorInfo GetOutputInfo(int index) override;

 private:
  /// \brief get the input and output tensor info from TensorRT engine
  void GetInputOutputInfo();
  /// \brief set input tensors to TensorRT engine
  void SetInputs(const std::vector<Tensor>& inputs);
  /// \brief allocate output tensors for TensorRT engine
  void AllocateOutputsBuffer(std::vector<Tensor>* outputs);

 private:
  /// \brief create TensorRT engine from serialized model file
  bool CreateTensorRTEngineFromSerialized();
  /// \brief create TensorRT engine from ONNX model file
  bool CreateTensorRTEngineFromONNX(const std::string& model_file);

  /// \brief load serialized TensorRT engine file
  bool LoadTensorRTEngineFromSerialized(const std::string& trt_engine_file);
  /// \brief load ONNX model file and serialize it to TensorRT engine file
  bool BuildTensorRTEngineFromFromONNX();

 private:
  /// \brief options for TensorRT inference backend
  TensorRTInferenceBackendOptions options_;
  /// \brief TensorRT engine
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  /// \brief TensorRT inference context
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  /// \brief TensorRT ONNX parser
  tensorrt::UniquePtr<nvonnxparser::IParser> parser_;
  /// \brief TensorRT engine builder
  tensorrt::UniquePtr<nvinfer1::IBuilder> builder_;
  /// \brief TensorRT network
  tensorrt::UniquePtr<nvinfer1::INetworkDefinition> network_;
  /// \brief CUDA stream for TensorRT inference
  cudaStream_t stream_{};
  /// \brief binding buffers for TensorRT inference
  std::vector<void*> bindings_;
  /// \brief input tensor info of TensorRT inference engine
  std::vector<TensorRTValueInfo> inputs_desc_;
  /// \brief output tensor info of TensorRT inference engine
  std::vector<TensorRTValueInfo> outputs_desc_;
  /// \brief input tensor buffers of TensorRT inference engine
  std::map<std::string, tensorrt::DeviceBuffer> inputs_buffer_;
  /// \brief output tensor buffers of TensorRT inference engine
  std::map<std::string, tensorrt::DeviceBuffer> outputs_buffer_;

  /// \brief the order of output tensors from TensorRT engine
  /// \details Sometimes while the number of outputs > 1, the output order of
  /// TensorRT may not be same with the original ONNX model. So this parameter
  /// will record to origin outputs order, to help recover the rigt order
  std::map<std::string, int> outputs_order_;
};

}  // namespace inference
}  // namespace zetton
