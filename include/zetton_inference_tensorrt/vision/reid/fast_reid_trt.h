#pragma once

#include "zetton_inference/interface/base_inference_model.h"
#include "zetton_inference/vision/base/matrix.h"
#include "zetton_inference/vision/base/result.h"

namespace zetton {
namespace inference {
namespace vision {

/// \brief FastReID TensorRT inference model
class FastReIDTensorRTInferenceModel : public BaseInferenceModel {
 public:
  /// \brief initialize the inference model with the given options
  bool Init(const InferenceRuntimeOptions& options) override;

  /// \brief do inference on the given input image
  /// \param im input image
  /// \param result output result
  bool Predict(cv::Mat* im, ReIDResult* result);

  /// \brief name of the inference model
  std::string Name() const override { return "FastReIDTensorRTInferenceModel"; }

 private:
  /// \brief Initialize the model, including the backend and other operations
  bool InitModel();

  /// \brief do preprocess for input image and return the preprocessed tensor
  /// \param im input image
  /// \param tensor preprocessed tensor
  /// \param im_info image info stored for postprocess
  bool Preprocess(Mat* mat, Tensor* output,
                  std::map<std::string, std::array<float, 2>>* im_info);

  /// \brief do postprocess for the backend inference output and return the
  /// result
  /// \param infer_result the backend inference output
  /// \param result the result of the model inference
  /// \param im_info the image information from preprocess
  /// \param conf_threshold the threshold of the confidence
  bool Postprocess(std::vector<Tensor>& infer_results, ReIDResult* result,
                   const std::map<std::string, std::array<float, 2>>& im_info);

 private:
  static std::vector<cv::Mat> CropSubImages(const cv::Mat& im,
                                            const DetectionResult& detections);
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
