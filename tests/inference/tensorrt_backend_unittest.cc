#include "zetton_inference_tensorrt/backend/tensorrt/tensorrt_backend.h"

#include <gtest/gtest.h>
#include <zetton_inference/base/options.h>
#include <zetton_inference/base/type.h>

TEST(TensorRTInferenceBackendTest, InitBackend) {
  auto backend =
      std::make_shared<zetton::inference::TensorRTInferenceBackend>();
  EXPECT_EQ(backend->Initialized(), false);

  zetton::inference::InferenceRuntimeOptions options;
  options.UseTensorRTBackend();
  options.UseGpu();
  options.model_format = zetton::inference::InferenceFrontendType::kSerialized;
  options.SetCacheFileForTensorRT(
      "/workspace/model/yolov3-tiny-416-bs1.engine");
  EXPECT_EQ(backend->Init(options), true);
  EXPECT_EQ(backend->Initialized(), true);
}
