
#pragma once

#include <NvInfer.h>

#include "zetton_common/log/log.h"

namespace zetton {
namespace inference {
namespace tensorrt {

class Logger : public nvinfer1::ILogger {
 public:
  static Logger* logger;
  static Logger* Get() {
    if (logger != nullptr) {
      return logger;
    }
    logger = new Logger();
    return logger;
  }
  void log(nvinfer1::ILogger::Severity severity,
           const char* msg) noexcept override {
    if (severity == nvinfer1::ILogger::Severity::kINFO) {
      // Disable this log
      // AINFO << msg << std::endl;
    } else if (severity == nvinfer1::ILogger::Severity::kWARNING) {
      // Disable this log
      // AWARN << msg << std::endl;
    } else if (severity == nvinfer1::ILogger::Severity::kERROR) {
      AERROR << msg << std::endl;
    } else if (severity == nvinfer1::ILogger::Severity::kINTERNAL_ERROR) {
      AFATAL << msg;
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace zetton
