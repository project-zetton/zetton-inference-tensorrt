#include "zetton_inference_tensorrt/vision/reid/fast_reid_trt.h"

namespace zetton {
namespace inference {
namespace vision {

bool FastReIDTensorRTInferenceModel::Init(
    const InferenceRuntimeOptions& options) {
  AERROR_F("Not implemented yet");
  return false;
}

bool FastReIDTensorRTInferenceModel::Predict(cv::Mat* im, ReIDResult* result) {
  AERROR_F("Not implemented yet");
  return false;
}

bool FastReIDTensorRTInferenceModel::InitModel() {
  AERROR_F("Not implemented yet");
  return false;
}

bool FastReIDTensorRTInferenceModel::Preprocess(
    Mat* mat, Tensor* output,
    std::map<std::string, std::array<float, 2>>* im_info) {
  AERROR_F("Not implemented yet");
  return false;
}

bool FastReIDTensorRTInferenceModel::Postprocess(
    std::vector<Tensor>& infer_results, ReIDResult* result,
    const std::map<std::string, std::array<float, 2>>& im_info) {
  AERROR_F("Not implemented yet");
  return false;
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
