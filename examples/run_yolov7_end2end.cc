#include <absl/strings/str_join.h>

#include <opencv2/imgcodecs.hpp>

#include "zetton_inference/vision/base/result.h"
#include "zetton_inference_tensorrt/vision/detection/yolov7_end2end_trt.h"

int main(int argc, char** argv) {
  // init options
  zetton::inference::InferenceRuntimeOptions options;
  options.UseTensorRTBackend();
  options.UseGpu();
#if 0
  options.model_format = zetton::inference::InferenceFrontendType::kSerialized;
  options.SetTensorRTCacheFile("/workspace/model/yolov7-tiny-nms.trt");
#else
  options.model_format = zetton::inference::InferenceFrontendType::kONNX;
  options.SetModelPath("/workspace/model/yolov7-tiny.onnx");
  options.SetTensorRTCacheFile(
      "/workspace/model/yolov7-tiny-nms-from-onnx.trt");
#endif

  // init detector
  auto detector = std::make_shared<
      zetton::inference::vision::YOLOv7End2EndTensorRTInferenceModel>();
  detector->Init(options);

  // load image
  cv::Mat image = cv::imread("/workspace/data/dog.jpg");

  // inference
  zetton::inference::vision::DetectionResult result;
  detector->Predict(&image, &result);

  // print result
  AINFO_F("Detected {} objects", result.boxes.size());
  for (std::size_t i = 0; i < result.boxes.size(); ++i) {
    AINFO_F("-> label: {}, score: {}, bbox: {} {} {} {}", result.label_ids[i],
            result.scores[i], result.boxes[i][0], result.boxes[i][1],
            result.boxes[i][2], result.boxes[i][3]);
  }

  // print benchmark
  zetton::inference::vision::DetectionResult temp_result;
  detector->EnableRecordTimeOfRuntime();
  for (auto i = 0; i < 100; ++i) {
    detector->Predict(&image, &temp_result);
  }
  detector->DisableRecordTimeOfRuntime();
  detector->PrintStatsInfoOfRuntime();

  return 0;
}