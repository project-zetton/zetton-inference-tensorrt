#include <memory>
#include <opencv2/imgcodecs.hpp>

#include "zetton_inference/vision/base/result.h"
#include "zetton_inference_tensorrt/vision/detection/yolov7_end2end_trt.h"
#include "zetton_inference_tensorrt/vision/reid/fast_reid_trt.h"

int main(int argc, char** argv) {
  // init detector
  zetton::inference::InferenceRuntimeOptions detector_options;
  detector_options.UseTensorRTBackend();
  detector_options.UseGpu();
  detector_options.model_format =
      zetton::inference::InferenceFrontendType::kSerialized;
  detector_options.SetCacheFileForTensorRT(
      "/workspace/model/yolov7-tiny-nms.trt");
  auto detector = std::make_shared<
      zetton::inference::vision::YOLOv7End2EndTensorRTInferenceModel>();
  detector->Init(detector_options,
                 zetton::inference::vision::YOLOEnd2EndModelType::kYOLOv7);

  // init feature extractor
  zetton::inference::InferenceRuntimeOptions extractor_options;
  extractor_options.UseTensorRTBackend();
  extractor_options.UseGpu();
  extractor_options.model_format =
      zetton::inference::InferenceFrontendType::kSerialized;
  extractor_options.SetCacheFileForTensorRT(
      "/workspace/model/market_bot_R50.trt");
  auto extractor = std::make_shared<
      zetton::inference::vision::FastReIDTensorRTInferenceModel>();
  extractor->Init(extractor_options);
  extractor->params.batch_size = 1;

  // load image
  cv::Mat image = cv::imread("/workspace/data/person.jpg");

  // detect objects
  zetton::inference::vision::DetectionResult detection_result;
  detector->Predict(&image, &detection_result, 0.25);

  // print detection result
  AINFO_F("Detected {} objects", detection_result.boxes.size());
  for (std::size_t i = 0; i < detection_result.boxes.size(); ++i) {
    AINFO_F("-> label: {}, score: {}, bbox: {} {} {} {}",
            detection_result.label_ids[i], detection_result.scores[i],
            detection_result.boxes[i][0], detection_result.boxes[i][1],
            detection_result.boxes[i][2], detection_result.boxes[i][3]);
  }

  // extract features
  zetton::inference::vision::ReIDResult extraction_result;
  extractor->Predict(&image, &detection_result, &extraction_result);

  // print result
  AINFO_F("Extract features from {} objects",
          extraction_result.features.size());
  for (int i = 0; i < extraction_result.features.size(); ++i) {
    AINFO_F("-> label {}: feature shape {}", detection_result.label_ids[i],
            extraction_result.features[i].size());
  }

  // print benchmark
  zetton::inference::vision::ReIDResult temp_result;
  extractor->EnableRecordTimeOfRuntime();
  for (auto i = 0; i < 100; ++i) {
    extractor->Predict(&image, &detection_result, &extraction_result);
  }
  extractor->DisableRecordTimeOfRuntime();
  extractor->PrintStatsInfoOfRuntime();

  return 0;
}
