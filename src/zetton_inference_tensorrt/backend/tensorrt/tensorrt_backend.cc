#include "zetton_inference_tensorrt/backend/tensorrt/tensorrt_backend.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <zetton_inference/base/type.h>

#include <fstream>

#include "zetton_common/log/log.h"
#include "zetton_inference/util/io_util.h"
#include "zetton_inference_tensorrt/backend/tensorrt/logging.h"
#include "zetton_inference_tensorrt/backend/tensorrt/util.h"

namespace zetton {
namespace inference {

tensorrt::Logger* tensorrt::Logger::logger = nullptr;

bool TensorRTInferenceBackend::Init(const InferenceRuntimeOptions& options) {
  options_.gpu_id = options.device_id;
  options_.enable_fp16 = options.trt_enable_fp16;
  options_.enable_int8 = options.trt_enable_int8;
  options_.max_batch_size = options.trt_max_batch_size;
  options_.max_workspace_size = options.trt_max_workspace_size;
  options_.max_shape = options.trt_max_shape;
  options_.min_shape = options.trt_min_shape;
  options_.opt_shape = options.trt_opt_shape;
  options_.serialize_file = options.trt_serialize_file;

  ACHECK_F(options.model_format == InferenceFrontendType::kSerialized ||
               options.model_format == InferenceFrontendType::kONNX,
           "Unsupported model format: {}", ToString(options.model_format));

  if (options.model_format == InferenceFrontendType::kSerialized) {
    ACHECK_F(InitFromSerialized(options_),
             "Failed to initialize TensorRT inference backend from serialized "
             "model file: {}",
             options_.serialize_file);
    return true;
  } else if (options.model_format == InferenceFrontendType::kONNX) {
    ACHECK_F(InitFromONNX(options.model_file, options_),
             "Failed to initialize TensorRT inference backend from ONNX model "
             "file: {}",
             options.model_file);
    return true;
  }

  return false;
}

bool TensorRTInferenceBackend::InitFromSerialized(
    const TensorRTInferenceBackendOptions& options) {
  if (initialized_) {
    AERROR_F("TensorRT inference backend has been initialized");
    return false;
  }
  options_ = options;
  cudaSetDevice(options_.gpu_id);

  ACHECK_F(cudaStreamCreate(&stream_) == 0,
           "Failed to create CUDA stream for TensorRT inference backend");

  if (!CreateTensorRTEngineFromSerialized()) {
    AERROR_F("Failed to create TensorRT engine from serialized model file: {}",
             options_.serialize_file);
    return false;
  }
  initialized_ = true;
  return true;
}

bool TensorRTInferenceBackend::InitFromONNX(
    const std::string& model_file,
    const TensorRTInferenceBackendOptions& options) {
  if (initialized_) {
    AERROR_F("TensorRT inference backend has been initialized");
    return false;
  }
  options_ = options;
  cudaSetDevice(options_.gpu_id);

  ACHECK_F(cudaStreamCreate(&stream_) == 0,
           "Failed to create CUDA stream for TensorRT inference backend");

  if (!CreateTensorRTEngineFromONNX(model_file)) {
    AERROR_F("Failed to create TensorRT engine from ONNX model file: {}",
             model_file);
    return false;
  }
  initialized_ = true;
  return true;
}

bool TensorRTInferenceBackend::CreateTensorRTEngineFromONNX(
    const std::string& model_file) {
  // 0. initialization
  // 0.1. init builder
  builder_.reset(nvinfer1::createInferBuilder(*tensorrt::Logger().Get()));
  if (!builder_) {
    AERROR_F("Failed to call createInferBuilder().");
    return false;
  }

  // 0.2. init network
  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  network_.reset(builder_->createNetworkV2(explicitBatch));
  if (!network_) {
    AERROR_F("Failed to call createNetworkV2().");
    return false;
  }

  // 0.3. init parser
  parser_.reset(
      nvonnxparser::createParser(*network_, *tensorrt::Logger().Get()));
  if (!parser_) {
    AERROR_F("Failed to call createParser().");
    return false;
  }
  if (!parser_->parseFromFile(
          model_file.c_str(),
          static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
    AERROR_F("Failed to parse ONNX model by TensorRT: {}", model_file);
  }

  // 1. check whether the model is already serialized
  if (options_.serialize_file != "") {
    std::ifstream fin(options_.serialize_file, std::ios::binary | std::ios::in);
    if (fin) {
      AINFO_F("Serialized model file already exists: {}, skip building engine",
              options_.serialize_file);
      fin.close();
      return LoadTensorRTEngineFromSerialized(options_.serialize_file);
    }
  }

  // 2. build engine if serialized file is not found
  if (!BuildTensorRTEngineFromFromONNX()) {
    AERROR_F("Failed to build TensorRT engine from ONNX model file: {}",
             model_file);
  }

  return true;
}

bool TensorRTInferenceBackend::BuildTensorRTEngineFromFromONNX() {
  // setup config
  auto config = builder_->createBuilderConfig();
#if NV_TENSORRT_MAJOR < 8
  config->setMaxWorkspaceSize(options_.max_workspace_size);
#else
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             options_.max_workspace_size);
#endif
  if (options_.enable_fp16) {
    if (!builder_->platformHasFastFp16()) {
      AWARN_F("FP16 is not supported on this platform, use FP32 instead");
    } else {
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
  }

  // setup profile
  auto profile = builder_->createOptimizationProfile();
  for (const auto& item : options_.min_shape) {
    ACHECK_F(profile->setDimensions(item.first.c_str(),
                                    nvinfer1::OptProfileSelector::kMIN,
                                    tensorrt::ToDims(item.second)),
             "Failed to set min shape for input: {}", item.first);
  }
  for (const auto& item : options_.max_shape) {
    ACHECK_F(profile->setDimensions(item.first.c_str(),
                                    nvinfer1::OptProfileSelector::kMAX,
                                    tensorrt::ToDims(item.second)),
             "Failed to set max shape for input: {}", item.first);
  }
  if (options_.opt_shape.empty()) {
    for (const auto& item : options_.max_shape) {
      ACHECK_F(profile->setDimensions(item.first.c_str(),
                                      nvinfer1::OptProfileSelector::kMAX,
                                      tensorrt::ToDims(item.second)),
               "Failed to set opt shape for input: {}", item.first);
    }
  } else {
    for (const auto& item : options_.opt_shape) {
      ACHECK_F(profile->setDimensions(item.first.c_str(),
                                      nvinfer1::OptProfileSelector::kOPT,
                                      tensorrt::ToDims(item.second)),
               "Failed to set opt shape for input: {}", item.first);
    }
  }
  config->addOptimizationProfile(profile);

  // build the engine
  AINFO_F("Building TensorRT engine, this will take a while...");
  if (context_) {
    context_.reset();
    engine_.reset();
  }
  builder_->setMaxBatchSize(options_.max_batch_size);

  tensorrt::UniquePtr<nvinfer1::IHostMemory> plan{
      builder_->buildSerializedNetwork(*network_, *config)};
  if (!plan) {
    AERROR_F("Failed to call buildSerializedNetwork().");
    return false;
  }

  // init runtime
  tensorrt::UniquePtr<nvinfer1::IRuntime> runtime{
      nvinfer1::createInferRuntime(*tensorrt::Logger::Get())};
  if (!runtime) {
    AERROR_F("Failed to call createInferRuntime().");
    return false;
  }

  // init engine
  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(plan->data(), plan->size()),
      tensorrt::InferDeleter());
  if (!engine_) {
    AERROR_F("Failed to call deserializeCudaEngine().");
    return false;
  }

  // init context
  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  if (!context_) {
    AERROR_F("Failed to call createExecutionContext().");
    return false;
  }
  GetInputOutputInfo();

  AINFO_F("TensorRT engine built successfully.");

  // save serialized engine
  if (!options_.serialize_file.empty()) {
    AINFO_F("Serialize TensorRT engine to local file: {}",
            options_.serialize_file);
    std::ofstream engine_file(options_.serialize_file.c_str());
    if (!engine_file) {
      AERROR_F("Failed to open {} to write.", options_.serialize_file);
      return false;
    }
    engine_file.write(static_cast<char*>(plan->data()), plan->size());
    engine_file.close();
    AINFO_F("Serialized TensorRT engine file saved to: {}",
            options_.serialize_file);
  }

  return true;
}

bool TensorRTInferenceBackend::CreateTensorRTEngineFromSerialized() {
  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  builder_ = tensorrt::UniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(*tensorrt::Logger::Get()));
  if (!builder_) {
    AERROR_F("Failed to call createInferBuilder().");
    return false;
  }
  network_ = tensorrt::UniquePtr<nvinfer1::INetworkDefinition>(
      builder_->createNetworkV2(explicitBatch));
  if (!network_) {
    AERROR_F("Failed to call createNetworkV2().");
    return false;
  }

  if (!options_.serialize_file.empty()) {
    std::ifstream fin(options_.serialize_file, std::ios::binary | std::ios::in);
    if (fin) {
      AINFO_F(
          "TensorRT serialized engine file found, deserialize from local file: "
          "{}",
          options_.serialize_file);
      fin.close();
      return LoadTensorRTEngineFromSerialized(options_.serialize_file);
    } else {
      AERROR_F("Failed to open serialized TensorRT Engine file: {}",
               options_.serialize_file);
    }
  } else {
    AFATAL_F("Serialized TensorRT engine file is not specified.");
  }

  return false;
}

bool TensorRTInferenceBackend::LoadTensorRTEngineFromSerialized(
    const std::string& trt_engine_file) {
  cudaSetDevice(options_.gpu_id);

  std::string engine_buffer;
  if (!ReadBinaryFromFile(trt_engine_file, &engine_buffer)) {
    AERROR_F("Failed to load TensorRT Engine from {}.", trt_engine_file);
    return false;
  }

  tensorrt::UniquePtr<nvinfer1::IRuntime> runtime{
      nvinfer1::createInferRuntime(*tensorrt::Logger::Get())};
  if (!runtime) {
    AERROR_F("Failed to call createInferRuntime().");
    return false;
  }

  // workaround to disable plugin initialization
  // refer to https://github.com/onnx/onnx-tensorrt/issues/597
  initLibNvInferPlugins(nullptr, "");

  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(engine_buffer.data(),
                                     engine_buffer.size()),
      tensorrt::InferDeleter());
  if (!engine_) {
    AERROR_F("Failed to call deserializeCudaEngine().");
    return false;
  }

  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  GetInputOutputInfo();

  AINFO_F("TensorRT engine loaded successfully.");

  return true;
}

bool TensorRTInferenceBackend::Infer(std::vector<Tensor>& inputs,
                                     std::vector<Tensor>* outputs) {
  if (static_cast<int>(inputs.size()) != NumInputs()) {
    AERROR_F("Number of inputs mismatch: {} vs {}", inputs.size(), NumInputs());
    return false;
  }

  SetInputs(inputs);
  AllocateOutputsBuffer(outputs);
  if (!context_->enqueueV2(bindings_.data(), stream_, nullptr)) {
    AERROR_F("Failed to infer with TensorRT.");
    return false;
  }
  for (auto& output : *outputs) {
    ACHECK_F(
        cudaMemcpyAsync(output.Data(), outputs_buffer_[output.name].data(),
                        output.Nbytes(), cudaMemcpyDeviceToHost, stream_) == 0,
        "Failed to copy output data from device to host.");
  }
  return true;
}

void TensorRTInferenceBackend::GetInputOutputInfo() {
  std::vector<TensorRTValueInfo>().swap(inputs_desc_);
  std::vector<TensorRTValueInfo>().swap(outputs_desc_);
  inputs_desc_.clear();
  outputs_desc_.clear();
  auto num_binds = engine_->getNbBindings();
  for (auto i = 0; i < num_binds; ++i) {
    std::string name = std::string(engine_->getBindingName(i));
    auto shape = tensorrt::ToVec(engine_->getBindingDimensions(i));
    auto dtype = engine_->getBindingDataType(i);
    if (engine_->bindingIsInput(i)) {
      inputs_desc_.emplace_back(TensorRTValueInfo{name, shape, dtype});
      inputs_buffer_[name] = tensorrt::DeviceBuffer(dtype);
    } else {
      outputs_desc_.emplace_back(TensorRTValueInfo{name, shape, dtype});
      outputs_buffer_[name] = tensorrt::DeviceBuffer(dtype);
    }
  }
  bindings_.resize(num_binds);
}

void TensorRTInferenceBackend::SetInputs(const std::vector<Tensor>& inputs) {
  for (const auto& item : inputs) {
    auto idx = engine_->getBindingIndex(item.name.c_str());
    std::vector<int> shape(item.shape.begin(), item.shape.end());
    auto dims = tensorrt::ToDims(shape);
    context_->setBindingDimensions(idx, dims);

    if (item.device == InferenceDeviceType::kGPU) {
      if (item.dtype == InferenceDataType::kINT64) {
        // TODO cast int64 to int32
        // TRT don't support INT64
        AFATAL_F("TensorRT don't support INT64, use INT32 instead.");
      } else {
        // no copy
        inputs_buffer_[item.name].SetExternalData(dims, item.Data());
      }
    } else {
      // Allocate input buffer memory
      inputs_buffer_[item.name].resize(dims);

      // copy from cpu to gpu
      if (item.dtype == InferenceDataType::kINT64) {
        int64_t* data = static_cast<int64_t*>(const_cast<void*>(item.Data()));
        std::vector<int32_t> casted_data(data, data + item.Numel());
        ACHECK_F(cudaMemcpyAsync(inputs_buffer_[item.name].data(),
                                 static_cast<void*>(casted_data.data()),
                                 item.Nbytes() / 2, cudaMemcpyHostToDevice,
                                 stream_) == 0,
                 "Failed to copy input data from host to device.");
      } else {
        ACHECK_F(cudaMemcpyAsync(inputs_buffer_[item.name].data(), item.Data(),
                                 item.Nbytes(), cudaMemcpyHostToDevice,
                                 stream_) == 0,
                 "Failed to copy input data from host to device.");
      }
    }
    // binding input buffer
    bindings_[idx] = inputs_buffer_[item.name].data();
  }
}

void TensorRTInferenceBackend::AllocateOutputsBuffer(
    std::vector<Tensor>* outputs) {
  if (outputs->size() != outputs_desc_.size()) {
    outputs->resize(outputs_desc_.size());
  }
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    auto idx = engine_->getBindingIndex(outputs_desc_[i].name.c_str());
    auto output_dims = context_->getBindingDimensions(idx);

    auto ori_idx = i;
    if (!outputs_order_.empty()) {
      // find the original index of output
      auto iter = outputs_order_.find(outputs_desc_[i].name);
      ACHECK_F(iter != outputs_order_.end(),
               "Failed to find the original index of output: {}.",
               outputs_desc_[i].name);
    }
    // set user's outputs info
    std::vector<int64_t> shape(output_dims.d,
                               output_dims.d + output_dims.nbDims);
    (*outputs)[ori_idx].Resize(
        shape, tensorrt::GetInferenceDataType(outputs_desc_[i].dtype),
        outputs_desc_[i].name);
    // Allocate output buffer memory
    outputs_buffer_[outputs_desc_[i].name].resize(output_dims);
    // binding output buffer
    bindings_[idx] = outputs_buffer_[outputs_desc_[i].name].data();
  }
}

TensorInfo TensorRTInferenceBackend::GetInputInfo(int index) {
  ACHECK_F(index < NumInputs(), "Input index {} is out of range: {}.", index,
           NumInputs());
  TensorInfo info;
  info.name = inputs_desc_[index].name;
  info.shape.assign(inputs_desc_[index].shape.begin(),
                    inputs_desc_[index].shape.end());
  info.dtype = tensorrt::GetInferenceDataType(inputs_desc_[index].dtype);
  return info;
}

TensorInfo TensorRTInferenceBackend::GetOutputInfo(int index) {
  ACHECK_F(index < NumOutputs(), "Output index {} is out of range: {}.", index,
           NumOutputs());
  TensorInfo info;
  info.name = outputs_desc_[index].name;
  info.shape.assign(outputs_desc_[index].shape.begin(),
                    outputs_desc_[index].shape.end());
  info.dtype = tensorrt::GetInferenceDataType(outputs_desc_[index].dtype);
  return info;
}

ZETTON_REGISTER_INFERENCE_BACKEND(TensorRTInferenceBackend)

}  // namespace inference
}  // namespace zetton
