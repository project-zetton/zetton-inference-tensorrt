#include <absl/strings/str_join.h>
#include <zetton_common/util/perf.h>

#include <opencv2/imgcodecs.hpp>

#include "zetton_inference/vision/base/result.h"
#include "zetton_inference/vision/util/visualize.h"
#include "zetton_inference_tensorrt/vision/detection/yolov7_end2end_trt.h"

int main(int argc, char** argv) {
  // init options
  zetton::inference::InferenceRuntimeOptions options;
  options.UseTensorRTBackend();
  options.UseGpu();
#if 1
  options.model_format = zetton::inference::InferenceFrontendType::kSerialized;
  options.SetCacheFileForTensorRT("/workspace/model/yolov7-tiny-nms.trt");
  // options.SetCacheFileForTensorRT("/workspace/model/yolov5n-nms.trt");
  // options.SetCacheFileForTensorRT("/workspace/model/yolox_s-nms.trt");
#else
  options.model_format = zetton::inference::InferenceFrontendType::kONNX;
  options.SetModelPath("/workspace/model/yolov7-tiny.onnx");
  options.SetCacheFileForTensorRT(
      "/workspace/model/yolov7-tiny-nms-from-onnx.trt");
#endif

  // init detector
  auto detector = std::make_shared<
      zetton::inference::vision::YOLOv7End2EndTensorRTInferenceModel>();
  detector->Init(options,
                 zetton::inference::vision::YOLOEnd2EndModelType::kYOLOv7);

  // load image
  cv::Mat image = cv::imread("/workspace/data/dog.jpg");

  // inference
  zetton::inference::vision::DetectionResult result;
  detector->Predict(&image, &result, 0.25);

  // print result
  AINFO_F("Detected {} objects", result.boxes.size());
  for (std::size_t i = 0; i < result.boxes.size(); ++i) {
    AINFO_F("-> label: {}, score: {}, bbox: {} {} {} {}", result.label_ids[i],
            result.scores[i], result.boxes[i][0], result.boxes[i][1],
            result.boxes[i][2], result.boxes[i][3]);
  }

  // draw and save result
  auto viz = zetton::inference::vision::Visualization();
  auto result_image = viz.Visualize(image, result);
  cv::imwrite("predictions.png", result_image);

  // print benchmark of inference time
  zetton::inference::vision::DetectionResult temp_result;
  detector->EnableRecordTimeOfRuntime();
  for (auto i = 0; i < 100; ++i) {
    detector->Predict(&image, &temp_result);
  }
  detector->DisableRecordTimeOfRuntime();
  detector->PrintStatsInfoOfRuntime();

  // print benchmark for total process time
  zetton::common::FpsCalculator fps;
  for (auto i = 0; i < 100; ++i) {
    fps.Start();
    detector->Predict(&image, &temp_result);
    fps.End();
  }
  fps.PrintInfo("YOLOv7 (total)");

  return 0;
}
