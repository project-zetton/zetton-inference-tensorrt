#pragma once

#include <string>

#include "zetton_inference/base/type.h"
#include "zetton_inference/interface/base_inference_model.h"
#include "zetton_inference/vision/base/matrix.h"
#include "zetton_inference/vision/base/result.h"

namespace zetton {
namespace inference {
namespace vision {

/// \brief types (variants) for YOLO End2End model
enum class YOLOEnd2EndModelType {
  kYOLOv5,
  kYOLOX,
  kYOLOv7,
};

/// \brief parameters for YOLO End2End model
struct YOLOEnd2EndModelParams {
  /// \brief input image size
  /// \note the input image will be resized to this size
  /// \details tuple of (width, height)
  std::vector<int> size = {640, 640};
  /// \brief padding value, size should be same with Channels
  std::vector<float> padding_value = {114.0f, 114.0f, 114.0f};
  /// \brief only pad to the minimum rectange which height and width is times of
  /// stride
  bool is_mini_pad = false;
  /// \brief is_mini_pad = false and is_no_pad = true, will resize the image to
  /// the set size
  bool is_no_pad = false;
  /// \brief if is_scale_up is false, the input image only can be zoom out, the
  /// maximum resize scale cannot exceed 1.0
  bool is_scale_up = false;
  /// \brief padding stride, for is_mini_pad
  int stride = 32;
};

/// \brief YOLOv7 end-to-end TensorRT inference model
class YOLOv7End2EndTensorRTInferenceModel : public BaseInferenceModel {
 public:
  /// \brief initialize the inference model with the given options
  /// \param options the options for the inference model
  bool Init(const InferenceRuntimeOptions& options) override;

  /// \brief initialize the inference model with the given options and model
  /// variant type
  /// \param options inference runtime options
  /// \param model_type model variant type
  bool Init(const InferenceRuntimeOptions& options,
            const YOLOEnd2EndModelType& model_type);

  /// \brief do inference on the given input image
  /// \param im input image
  /// \param result output result
  /// \param conf_threshold confidence threshold
  bool Predict(cv::Mat* im, DetectionResult* result,
               float conf_threshold = 0.25);

  /// \brief name of the inference model
  std::string Name() const override {
    return "YOLOv7End2EndTensorRTInferenceModel";
  }

 public:
  /// \brief parameters for the model
  YOLOEnd2EndModelParams params;

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
  bool Postprocess(std::vector<Tensor>& infer_results, DetectionResult* result,
                   const std::map<std::string, std::array<float, 2>>& im_info,
                   float conf_threshold);

 private:
  /// \brief do letterbox for the input image
  /// \details resize the image to the set size and pad the image to the minimum
  /// \param mat the input image
  /// \param size the set size
  void LetterBox(Mat* mat, const std::vector<int>& size,
                 const std::vector<float>& color, bool _auto,
                 bool scale_fill = false, bool scale_up = true,
                 int stride = 32);

 private:
  /// \brief whether or not the input batch size is dynamic
  bool is_dynamic_input_;

  /// \brief model variant type
  YOLOEnd2EndModelType model_type_ = YOLOEnd2EndModelType::kYOLOv7;
};

}  // namespace vision
}  // namespace inference
}  // namespace zetton
