#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

#include <opencv2/opencv.hpp>

#include "zetton_common/util/perf.h"
#include "zetton_inference/vision/base/result.h"
#include "zetton_inference/vision/tracking/bytetrack/byte_tracker.h"
#include "zetton_inference/vision/util/visualize.h"
#include "zetton_inference_tensorrt/vision/detection/yolov7_end2end_trt.h"

ABSL_FLAG(std::string, input_file,
          "/workspace/data/sample-videos/person-bicycle-car-detection.mp4",
          "path to input video file");
ABSL_FLAG(std::string, output_file, "result.avi", "path to output video file");
ABSL_FLAG(std::string, detection_model_path,
          "/workspace/model/yolov7-tiny-nms.trt",
          "path to YOLOv7 detection model file");

int main(int argc, char** argv) {
  // parse args
  absl::ParseCommandLine(argc, argv);
  auto input_file = absl::GetFlag(FLAGS_input_file);
  auto output_file = absl::GetFlag(FLAGS_output_file);
  auto detection_model_path = absl::GetFlag(FLAGS_detection_model_path);

  // init detector
  zetton::inference::InferenceRuntimeOptions detector_options;
  detector_options.UseTensorRTBackend();
  detector_options.UseGpu();
  detector_options.model_format =
      zetton::inference::InferenceFrontendType::kSerialized;
  detector_options.SetCacheFileForTensorRT(detection_model_path);
  auto detector = std::make_shared<
      zetton::inference::vision::YOLOv7End2EndTensorRTInferenceModel>();
  detector->Init(detector_options,
                 zetton::inference::vision::YOLOEnd2EndModelType::kYOLOv7);

  // init tracker
  zetton::inference::vision::ByteTrackerParams tracker_params;
  zetton::inference::vision::ByteTracker tracker;
  tracker.Init(tracker_params);

  // init video IO
  AINFO_F("Processing: {}", input_file);
  cv::VideoCapture video_cap(input_file);
  cv::Size sSize = cv::Size((int)video_cap.get(cv::CAP_PROP_FRAME_WIDTH),
                            (int)video_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  auto fFps = static_cast<float>(video_cap.get(cv::CAP_PROP_FPS));
  AINFO_F("Frame width is: {}, height is: {}, video fps is: {}", sSize.width,
          sSize.height, fFps);
  cv::VideoWriter video_writer(
      output_file, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fFps, sSize);

  // do detection and tracking
  cv::Mat src_img;
  zetton::inference::vision::DetectionResult detection_result;
  zetton::inference::vision::TrackingResult tracking_result;
  zetton::inference::vision::Visualization viz;
  zetton::common::FpsCalculator fps;
  while (video_cap.read(src_img)) {
    // start timer
    fps.Start();
    // detect objects
    detector->Predict(&src_img, &detection_result, 0.25);
    AINFO_F("Detected {} objects", detection_result.boxes.size());
    for (std::size_t i = 0; i < detection_result.boxes.size(); ++i) {
      AINFO_F("-> label: {}, score: {}, bbox: {} {} {} {}",
              detection_result.label_ids[i], detection_result.scores[i],
              detection_result.boxes[i][0], detection_result.boxes[i][1],
              detection_result.boxes[i][2], detection_result.boxes[i][3]);
    }
    // track objects
    tracker.Update(detection_result, tracking_result);
    AINFO_F("Tracked {} objects", tracking_result.boxes.size());
    for (std::size_t i = 0; i < tracking_result.boxes.size(); ++i) {
      AINFO_F("-> label: {}, score: {}, bbox: {} {} {} {}",
              tracking_result.label_ids[i], tracking_result.scores[i],
              tracking_result.boxes[i][0], tracking_result.boxes[i][1],
              tracking_result.boxes[i][2], tracking_result.boxes[i][3]);
    }
    // stop timer
    fps.End();
    // draw tracking results
    auto result_img = viz.Visualize(src_img, tracking_result);
    // write to output video
    video_writer.write(result_img);

    AINFO_F("--------------------");
  }

  // print profiling info
  fps.PrintInfo("YOLOv7 w/ ByteTrack");

  // print other messages
  AINFO_F("Result video saved to: {}", output_file);
  AINFO_F("Done!");

  return 0;
}
