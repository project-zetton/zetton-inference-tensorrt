#include "zetton_inference_tensorrt/backend/tensorrt/tensorrt_backend.h"

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
  auto tensorrt_options = TensorRTInferenceBackendOptions();
  tensorrt_options.gpu_id = options.device_id;
  tensorrt_options.enable_fp16 = options.trt_enable_fp16;
  tensorrt_options.enable_int8 = options.trt_enable_int8;
  tensorrt_options.max_batch_size = options.trt_max_batch_size;
  tensorrt_options.max_workspace_size = options.trt_max_workspace_size;
  tensorrt_options.max_shape = options.trt_max_shape;
  tensorrt_options.min_shape = options.trt_min_shape;
  tensorrt_options.opt_shape = options.trt_opt_shape;
  tensorrt_options.serialize_file = options.trt_serialize_file;

  ACHECK_F(options.model_format == InferenceFrontendType::kSerialized ||
               options.model_format == InferenceFrontendType::kONNX,
           "TrtBackend only support model format of {} and {}.",
           ToString(InferenceFrontendType::kSerialized),
           ToString(InferenceFrontendType::kONNX));

  if (options.model_format == InferenceFrontendType::kSerialized) {
    ACHECK_F(InitFromSerialized(tensorrt_options),
             "Load model from TensorRT Engine failed while initliazing "
             "TensorRTInferenceBackend.");
    return true;
  } else if (options.model_format == InferenceFrontendType::kONNX) {
    AFATAL_F("Not implemented yet.");
    // ACHECK_F(InitFromOnnx(options.model_file, trt_option),
    //          "Load model from ONNX failed while initliazing "
    //          "TensorRTInferenceBackend.");
    // return true;
    return false;
  }

  return false;
}

bool TensorRTInferenceBackend::InitFromSerialized(
    const TensorRTInferenceBackendOptions& options) {
  if (initialized_) {
    AERROR_F("TrtBackend is already initlized, cannot initialize again.");
    return false;
  }
  options_ = options;
  cudaSetDevice(options_.gpu_id);

  ACHECK_F(cudaStreamCreate(&stream_) == 0,
           "[ERROR] Error occurs while calling cudaStreamCreate().");

  if (!CreateTrtEngine()) {
    AERROR_F("Failed to create tensorrt engine.");
    return false;
  }
  initialized_ = true;
  return true;
}

bool TensorRTInferenceBackend::CreateTrtEngine() {
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
          "Detect serialized TensorRT Engine file in {}, will load it "
          "directly.",
          options_.serialize_file);
      fin.close();
      return LoadTrtCache(options_.serialize_file);
    } else {
      AERROR_F("Failed to open serialized TensorRT Engine file: {}",
               options_.serialize_file);
    }
  } else {
    AFATAL_F(
        "No serialized TensorRT Engine file is provided. Other methods are "
        "not supported yet.");
  }

  return false;
}

bool TensorRTInferenceBackend::LoadTrtCache(
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

  AINFO_F("Build TensorRT Engine from cache file {}", trt_engine_file);

  return true;
}

bool TensorRTInferenceBackend::Infer(std::vector<Tensor>& inputs,
                                     std::vector<Tensor>* outputs) {
  if (inputs.size() != NumInputs()) {
    AERROR_F("Require {} inputs, but get {}.", NumInputs(), inputs.size());
    return false;
  }

  SetInputs(inputs);
  AllocateOutputsBuffer(outputs);
  if (!context_->enqueueV2(bindings_.data(), stream_, nullptr)) {
    AERROR_F("Failed to Infer with TensorRT.");
    return false;
  }
  for (size_t i = 0; i < outputs->size(); ++i) {
    ACHECK_F(cudaMemcpyAsync((*outputs)[i].Data(),
                             outputs_buffer_[(*outputs)[i].name].data(),
                             (*outputs)[i].Nbytes(), cudaMemcpyDeviceToHost,
                             stream_) == 0,
             "Error occurs while copy memory from GPU to CPU.");
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
        AFATAL_F(
            "TRT don't support INT64 input on GPU, please use INT32 input");
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
                 "Error occurs while copy memory from CPU to GPU.");
      } else {
        ACHECK_F(cudaMemcpyAsync(inputs_buffer_[item.name].data(), item.Data(),
                                 item.Nbytes(), cudaMemcpyHostToDevice,
                                 stream_) == 0,
                 "Error occurs while copy memory from CPU to GPU.");
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
      ACHECK_F(
          iter != outputs_order_.end(),
          "Cannot find output: {} of tensorrt network from the original model.",
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
  ACHECK_F(index < NumInputs(),
           "The index: %d should less than the number of inputs: %d.", index,
           NumInputs());
  TensorInfo info;
  info.name = inputs_desc_[index].name;
  info.shape.assign(inputs_desc_[index].shape.begin(),
                    inputs_desc_[index].shape.end());
  info.dtype = tensorrt::GetInferenceDataType(inputs_desc_[index].dtype);
  return info;
}

TensorInfo TensorRTInferenceBackend::GetOutputInfo(int index) {
  ACHECK_F(index < NumOutputs(),
           "The index: %d should less than the number of outputs: %d.", index,
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
