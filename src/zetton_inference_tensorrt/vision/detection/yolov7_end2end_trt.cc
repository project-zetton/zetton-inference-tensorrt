#include "zetton_inference_tensorrt/vision/detection/yolov7_end2end_trt.h"

#include <zetton_inference/base/options.h>
#include <zetton_inference/base/type.h>

#include <opencv2/imgproc.hpp>

#include "zetton_inference/vision/base/result.h"
#include "zetton_inference/vision/base/transform/cast.h"
#include "zetton_inference/vision/base/transform/color_space_convert.h"
#include "zetton_inference/vision/base/transform/interpolate.h"
#include "zetton_inference/vision/base/transform/normalize.h"
#include "zetton_inference/vision/base/transform/pad.h"
#include "zetton_inference/vision/base/transform/permute.h"

namespace zetton {
namespace inference {
namespace vision {

void YOLOv7End2EndTensorRTInferenceModel::LetterBox(
    Mat* mat, const std::vector<int>& size, const std::vector<float>& color,
    bool _auto, bool scale_fill, bool scale_up, int stride) {
  float scale =
      std::min(size[1] * 1.0 / mat->Height(), size[0] * 1.0 / mat->Width());
  if (!scale_up) {
    scale = std::min(scale, 1.0f);
  }

  int resize_h = int(round(mat->Height() * scale));
  int resize_w = int(round(mat->Width() * scale));

  int pad_w = size[0] - resize_w;
  int pad_h = size[1] - resize_h;
  if (_auto) {
    pad_h = pad_h % stride;
    pad_w = pad_w % stride;
  } else if (scale_fill) {
    pad_h = 0;
    pad_w = 0;
    resize_h = size[1];
    resize_w = size[0];
  }
  if (resize_h != mat->Height() || resize_w != mat->Width()) {
    Interpolate::Run(mat, resize_w, resize_h);
  }
  if (pad_h > 0 || pad_w > 0) {
    float half_h = pad_h * 1.0 / 2;
    int top = int(round(half_h - 0.1));
    int bottom = int(round(half_h + 0.1));
    float half_w = pad_w * 1.0 / 2;
    int left = int(round(half_w - 0.1));
    int right = int(round(half_w + 0.1));
    Pad::Run(mat, top, bottom, left, right, color);
  }
}

bool YOLOv7End2EndTensorRTInferenceModel::Init(
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

bool YOLOv7End2EndTensorRTInferenceModel::Init(
    const InferenceRuntimeOptions& options,
    const YOLOEnd2EndModelType& model_type) {
  auto ret = Init(options);
  model_type_ = model_type;
  return ret;
}

bool YOLOv7End2EndTensorRTInferenceModel::InitModel() {
  // init parameters
  // ...

  // init inference runtime
  if (!InitRuntime()) {
    AERROR_F("Failed to initialize runtime.");
    return false;
  }

  // Check if the input shape is dynamic after Runtime already initialized,
  // Note that, We need to force is_mini_pad 'false' to keep static
  // shape after padding (LetterBox) when the is_dynamic_shape is 'false'.
  is_dynamic_input_ = false;
  auto shape = InputInfoOfRuntime(0).shape;
  for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
    // if height or width is dynamic
    if (i >= 2 && shape[i] <= 0) {
      is_dynamic_input_ = true;
      break;
    }
  }
  if (!is_dynamic_input_) {
    params.is_mini_pad = false;
  }
  return true;
}

bool YOLOv7End2EndTensorRTInferenceModel::Preprocess(
    Mat* mat, Tensor* output,
    std::map<std::string, std::array<float, 2>>* im_info) {
  float ratio =
      std::min(params.size[1] * 1.0f / static_cast<float>(mat->Height()),
               params.size[0] * 1.0f / static_cast<float>(mat->Width()));
  if (ratio != 1.0) {
    int interp = cv::INTER_AREA;
    if (ratio > 1.0) {
      interp = cv::INTER_LINEAR;
    }
    int resize_h = int(mat->Height() * ratio);
    int resize_w = int(mat->Width() * ratio);
    Interpolate::Run(mat, resize_w, resize_h, -1, -1, interp);
  }
  LetterBox(mat, params.size, params.padding_value, params.is_mini_pad,
            params.is_no_pad, params.is_scale_up, params.stride);

  if (model_type_ == YOLOEnd2EndModelType::kYOLOX) {
    // not do BGR2RGB and Normalize when using YOLOX models, or we will get
    // empty predictions according to [this
    // issue](https://github.com/Linaom1214/TensorRT-For-YOLO-Series/issues/11)
  } else {
    BGR2RGB::Run(mat);
    std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    std::vector<float> beta = {0.0f, 0.0f, 0.0f};
    Normalize::Run(mat, alpha, beta);
  }

  (*im_info)["output_shape"] = {static_cast<float>(mat->Height()),
                                static_cast<float>(mat->Width())};

  HWC2CHW::Run(mat);
  Cast::Run(mat, "float");
  mat->ShareWithTensor(output);
  output->shape.insert(output->shape.begin(), 1);  // reshape to n, h, w, c
  return true;
}

bool YOLOv7End2EndTensorRTInferenceModel::Postprocess(
    std::vector<Tensor>& infer_results, DetectionResult* result,
    const std::map<std::string, std::array<float, 2>>& im_info,
    float conf_threshold) {
  ACHECK_F(infer_results.size() == 4, "Output tensor size must be 4.");
  Tensor& num_tensor = infer_results.at(0);      // INT32
  Tensor& boxes_tensor = infer_results.at(1);    // FLOAT
  Tensor& scores_tensor = infer_results.at(2);   // FLOAT
  Tensor& classes_tensor = infer_results.at(3);  // INT32
  ACHECK_F(num_tensor.dtype == InferenceDataType::kINT32,
           "The dtype of num_dets must be INT32.");
  ACHECK_F(boxes_tensor.dtype == InferenceDataType::kFP32,
           "The dtype of det_boxes_tensor must be FP32.");
  ACHECK_F(scores_tensor.dtype == InferenceDataType::kFP32,
           "The dtype of det_scores_tensor must be FP32.");
  ACHECK_F(classes_tensor.dtype == InferenceDataType::kINT32,
           "The dtype of det_classes_tensor must be INT32.");
  ACHECK_F(num_tensor.shape[0] == 1, "Only support batch=1 now.");
  // post-process for end2end yolov7 after trt nms.
  auto* boxes_data = static_cast<float*>(boxes_tensor.Data());    // (1,100,4)
  auto* scores_data = static_cast<float*>(scores_tensor.Data());  // (1,100)
  auto* classes_data = static_cast<int32_t*>(classes_tensor.Data());  // (1,100)
  int32_t num_dets_after_trt_nms = static_cast<int32_t*>(num_tensor.Data())[0];
  result->Clear();
  result->Reserve(num_dets_after_trt_nms);
  if (num_dets_after_trt_nms == 0) {
    return true;
  }
  for (int i = 0; i < num_dets_after_trt_nms; ++i) {
    float confidence = scores_data[i];
    if (confidence <= conf_threshold) {
      continue;
    }
    int32_t label_id = classes_data[i];
    float x1 = boxes_data[(i * 4) + 0];
    float y1 = boxes_data[(i * 4) + 1];
    float x2 = boxes_data[(i * 4) + 2];
    float y2 = boxes_data[(i * 4) + 3];

    result->boxes.emplace_back(std::array<float, 4>{x1, y1, x2, y2});
    result->label_ids.push_back(label_id);
    result->scores.push_back(confidence);
  }

  if (result->boxes.size() == 0) {
    return true;
  }

  // scale the boxes to the origin image shape
  auto iter_out = im_info.find("output_shape");
  auto iter_ipt = im_info.find("input_shape");
  ACHECK_F(iter_out != im_info.end() && iter_ipt != im_info.end(),
           "Failed to find output_shape or input_shape in im_info.");
  float out_h = iter_out->second[0];
  float out_w = iter_out->second[1];
  float ipt_h = iter_ipt->second[0];
  float ipt_w = iter_ipt->second[1];
  float scale = std::min(out_h / ipt_h, out_w / ipt_w);
  float pad_h = (out_h - ipt_h * scale) / 2.0f;
  float pad_w = (out_w - ipt_w * scale) / 2.0f;
  if (params.is_mini_pad) {
    pad_h = static_cast<float>(static_cast<int>(pad_h) % params.stride);
    pad_w = static_cast<float>(static_cast<int>(pad_w) % params.stride);
  }
  for (auto& boxe : result->boxes) {
    // int32_t label_id = (result->label_ids)[i];
    boxe[0] = std::max((boxe[0] - pad_w) / scale, 0.0f);
    boxe[1] = std::max((boxe[1] - pad_h) / scale, 0.0f);
    boxe[2] = std::max((boxe[2] - pad_w) / scale, 0.0f);
    boxe[3] = std::max((boxe[3] - pad_h) / scale, 0.0f);
    boxe[0] = std::min(boxe[0], ipt_w - 1.0f);
    boxe[1] = std::min(boxe[1], ipt_h - 1.0f);
    boxe[2] = std::min(boxe[2], ipt_w - 1.0f);
    boxe[3] = std::min(boxe[3], ipt_h - 1.0f);
  }
  return true;
}

bool YOLOv7End2EndTensorRTInferenceModel::Predict(cv::Mat* im,
                                                  DetectionResult* result,
                                                  float conf_threshold) {
  Mat mat(*im);
  std::vector<Tensor> input_tensors(1);

  std::map<std::string, std::array<float, 2>> im_info;

  // Record the shape of image and the shape of preprocessed image
  im_info["input_shape"] = {static_cast<float>(mat.Height()),
                            static_cast<float>(mat.Width())};
  im_info["output_shape"] = {static_cast<float>(mat.Height()),
                             static_cast<float>(mat.Width())};

  if (!Preprocess(&mat, &input_tensors[0], &im_info)) {
    AERROR_F("Failed to preprocess input image.");
    return false;
  }

  input_tensors[0].name = InputInfoOfRuntime(0).name;
  std::vector<Tensor> output_tensors;
  if (!Infer(input_tensors, &output_tensors)) {
    AERROR_F("Failed to do inference.");
    return false;
  }

  if (!Postprocess(output_tensors, result, im_info, conf_threshold)) {
    AERROR_F("Failed to do post process.");
    return false;
  }

  return true;
}

}  // namespace vision
}  // namespace inference
}  // namespace zetton
