#include "zetton_inference_tensorrt/vision/reid/fast_reid_trt.h"

#include <cstddef>
#include <opencv2/imgproc.hpp>

#include "zetton_inference/vision/base/transform/cast.h"
#include "zetton_inference/vision/base/transform/interpolate.h"
#include "zetton_inference/vision/base/transform/permute.h"

namespace zetton {
namespace inference {
namespace vision {

bool FastReIDTensorRTInferenceModel::Init(
    const InferenceRuntimeOptions& options) {
  runtime_options = options;

  // check inference frontend
  if (runtime_options.model_format == InferenceFrontendType::kONNX ||
      runtime_options.model_format == InferenceFrontendType::kSerialized) {
    valid_cpu_backends = {};                                 // NO CPU
    valid_gpu_backends = {InferenceBackendType::kTensorRT};  // NO ORT
  }

  // check inference device
  if (runtime_options.device != InferenceDeviceType::kGPU) {
    AWARN_F("{} is not support for {}, will fallback to {}.",
            ToString(runtime_options.device), Name(),
            ToString(InferenceDeviceType::kGPU));
    runtime_options.device = InferenceDeviceType::kGPU;
  }

  // check inference backend
  if (runtime_options.backend != InferenceBackendType::kUnknown) {
    if (runtime_options.backend != InferenceBackendType::kTensorRT) {
      AWARN_F("{} is not support for {}, will fallback to {}.",
              ToString(runtime_options.backend), Name(),
              ToString(InferenceBackendType::kTensorRT));
      runtime_options.backend = InferenceBackendType::kTensorRT;
    }
  }

  // initialize inference model
  initialized = InitModel();

  return initialized;
}

bool FastReIDTensorRTInferenceModel::Predict(cv::Mat* im,
                                             DetectionResult* detections,
                                             ReIDResult* result) {
  // crop image by deteciton result
  auto sub_images = CropSubImages(*im, *detections);

  // do inference on each sub image
  int start_index = 0;
  int end_index = 0;
  while (end_index < static_cast<int>(sub_images.size())) {
    // get the sub images for current batch
    end_index =
        std::min(int(start_index + params.batch_size), int(sub_images.size()));
    auto iter_1 = sub_images.begin() + start_index;
    auto iter_2 = sub_images.begin() + end_index;
    std::vector<cv::Mat> single_batch_images(iter_1, iter_2);

    // do inference on current batch
    ReIDResult single_batch_result;
    if (!PredictSingleBatch(&single_batch_images, &single_batch_result)) {
      AERROR_F("Failed to do inference on batch {}-{}", start_index, end_index);
      return false;
    }

    // append the result of current batch to the final result
    result->features.insert(result->features.end(),
                            single_batch_result.features.begin(),
                            single_batch_result.features.end());

    start_index += params.batch_size;
  }

  return true;
}

bool FastReIDTensorRTInferenceModel::PredictSingleBatch(
    std::vector<cv::Mat>* single_batch_images, ReIDResult* result) {
  // prepare input tensor
  std::vector<Tensor> input_tensors(1);
  input_tensors[0].Resize({static_cast<int>(single_batch_images->size()),
                           params.channels, params.size[1], params.size[0]});

  // prepare image info
  std::map<std::string, std::array<float, 2>> im_info;
  im_info["current_batch_size"] = {
      static_cast<float>(single_batch_images->size()), 0.0f};

  // TODO: add multi-thread support
  // do preprocess for each sub image
  std::vector<Tensor> preprocessed_tensors;
  for (int i = 0; i < static_cast<int>(single_batch_images->size()); ++i) {
    Mat temp_mat((*single_batch_images)[i]);
    Tensor temp_tensor;
    if (!Preprocess(&temp_mat, &temp_tensor, &im_info)) {
      AERROR_F("Failed to preprocess input image.");
      return false;
    }
    preprocessed_tensors.push_back(temp_tensor);

    // TODO: avoid memory copy
    // copy preprocessed tensors to input tensor
    memcpy(reinterpret_cast<char*>(input_tensors[0].MutableData()) +
               static_cast<ptrdiff_t>(i * params.channels * params.size[1] *
                                      params.size[0]),
           preprocessed_tensors[i].Data(),
           params.channels * params.size[1] * params.size[0]);
  }

  // do inference
  input_tensors[0].name = InputInfoOfRuntime(0).name;
  std::vector<Tensor> output_tensors;
  if (!Infer(input_tensors, &output_tensors)) {
    AERROR_F("Failed to do inference.");
    return false;
  }

  // do postprocess
  if (!Postprocess(output_tensors, result, im_info)) {
    AERROR_F("Failed to do post process.");
    return false;
  }

  return true;
}

bool FastReIDTensorRTInferenceModel::InitModel() {
  // init parameters
  // ...

  // init inference runtime
  if (!InitRuntime()) {
    AERROR_F("Failed to initialize runtime.");
    return false;
  }

  // extra checks
  // ...

  return true;
}

bool FastReIDTensorRTInferenceModel::Preprocess(
    Mat* mat, Tensor* output,
    std::map<std::string, std::array<float, 2>>* im_info) {
  // resize
  float ratio =
      std::min(params.size[1] * 1.0f / static_cast<float>(mat->Height()),
               params.size[0] * 1.0f / static_cast<float>(mat->Width()));
  int interp = cv::INTER_AREA;
  if (ratio > 1.0) {
    interp = cv::INTER_LINEAR;
  }
  Interpolate::Run(mat, params.size[0], params.size[1], -1, -1, interp);

  HWC2CHW::Run(mat);

  Cast::Run(mat, "float");

  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);  // reshape to n, h, w, c
  return true;
}

bool FastReIDTensorRTInferenceModel::Postprocess(
    std::vector<Tensor>& infer_results, ReIDResult* result,
    const std::map<std::string, std::array<float, 2>>& im_info) {
  // get output features
  ACHECK_F(infer_results.size() == 1, "Output tensor size must be 1.");
  Tensor& feature_tensor = infer_results.at(0);
  ACHECK_F(feature_tensor.dtype == InferenceDataType::kFP32,
           "The dtype of det_scores_tensor must be FP32.");
  auto* feature_data = static_cast<float*>(feature_tensor.Data());

  // reshape output to result
  auto iter_batch_size = im_info.find("current_batch_size");
  if (iter_batch_size == im_info.end()) {
    AERROR_F("Failed to find current_batch_size in im_info.");
    return false;
  }
  auto current_batch_size = static_cast<int>(iter_batch_size->second[0]);
  auto feature_size = feature_tensor.Numel() / current_batch_size;
  result->Clear();
  result->Reserve(current_batch_size);
  for (int i = 0; i < current_batch_size; ++i) {
    result->features.emplace_back(
        feature_data + static_cast<ptrdiff_t>(i * feature_size),
        feature_data + static_cast<ptrdiff_t>((i + 1) * feature_size));
    // normalize the feature vector
    float norm = 0.0f;
    for (auto& f : result->features.back()) {
      norm += f * f;
    }
    norm = std::sqrt(norm);
    for (auto& f : result->features.back()) {
      f /= norm;
    }
  }

  return true;
}

std::vector<cv::Mat> FastReIDTensorRTInferenceModel::CropSubImages(
    const cv::Mat& im, const DetectionResult& detections) {
  std::vector<cv::Mat> sub_images;
  for (const auto& bbox : detections.boxes) {
    int x1 = std::max(static_cast<int>(bbox[0]), 0);
    int y1 = std::max(static_cast<int>(bbox[1]), 0);
    int x2 = std::min(static_cast<int>(bbox[2]), im.cols);
    int y2 = std::min(static_cast<int>(bbox[3]), im.rows);
    cv::Rect rect{x1, y1, x2 - x1, y2 - y1};
    cv::Mat sub_image = im(rect);
    sub_images.push_back(sub_image);
  }
  return sub_images;
}

}  // namespace vision
}  // namespace inference
}  // namespace zetton
