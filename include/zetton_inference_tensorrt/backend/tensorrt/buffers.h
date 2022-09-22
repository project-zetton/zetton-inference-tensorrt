#pragma once

#include <NvInfer.h>

#include <cstdlib>
#include <new>

#include "zetton_inference_tensorrt/backend/tensorrt/common.h"

namespace zetton {
namespace inference {
namespace tensorrt {

/// \brief  The GenericBuffer class is a templated class for buffers.
///
/// \details This templated RAII (Resource Acquisition Is Initialization) class
/// handles the allocation, deallocation, querying of buffers on both the device
/// and the host. It can handle data of arbitrary types because it stores byte
/// buffers. The template parameters AllocFunc and FreeFunc are used for the
/// allocation and deallocation of the buffer. AllocFunc must be a functor that
/// takes in (void** ptr, size_t size) and returns bool. ptr is a pointer to
/// where the allocated buffer address should be stored. size is the amount of
/// memory in bytes to allocate. The boolean indicates whether or not the memory
/// allocation was successful. FreeFunc must be a functor that takes in (void*
/// ptr) and returns void. ptr is the allocated buffer address. It must work
/// with nullptr input.
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer {
 public:
  /// \brief Construct an empty buffer.
  GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
      : mType(type), mBuffer(nullptr) {}

  /// \brief Construct a buffer with the specified allocation size in bytes.
  GenericBuffer(size_t size, nvinfer1::DataType type)
      : mSize(size), mCapacity(size), mType(type) {
    if (!allocFn(&mBuffer, this->nbBytes())) {
      throw std::bad_alloc();
    }
  }

  /// \brief This use to skip memory copy step.
  GenericBuffer(size_t size, nvinfer1::DataType type, void* buffer)
      : mSize(size), mCapacity(size), mType(type) {
    mExternal_buffer = buffer;
  }

  GenericBuffer(GenericBuffer&& buf) noexcept
      : mSize(buf.mSize),
        mCapacity(buf.mCapacity),
        mType(buf.mType),
        mBuffer(buf.mBuffer),
        mExternal_buffer(nullptr) {
    buf.mSize = 0;
    buf.mCapacity = 0;
    buf.mType = nvinfer1::DataType::kFLOAT;
    buf.mBuffer = nullptr;
  }

  GenericBuffer& operator=(GenericBuffer&& buf) noexcept {
    if (this != &buf) {
      freeFn(mBuffer);
      mSize = buf.mSize;
      mCapacity = buf.mCapacity;
      mType = buf.mType;
      mBuffer = buf.mBuffer;
      // Reset buf.
      buf.mSize = 0;
      buf.mCapacity = 0;
      buf.mBuffer = nullptr;
    }
    return *this;
  }

  /// \brief Returns pointer to underlying array.
  void* data() {
    if (mExternal_buffer != nullptr) {
      return mExternal_buffer;
    }
    return mBuffer;
  }

  /// \brief Returns pointer to underlying array.
  const void* data() const {
    if (mExternal_buffer != nullptr) {
      return mExternal_buffer;
    }
    return mBuffer;
  }

  /// \brief Returns the size (in number of elements) of the buffer.
  size_t size() const { return mSize; }

  /// \brief Returns the size (in bytes) of the buffer.
  size_t nbBytes() const { return this->size() * getElementSize(mType); }

  /// \brief Resizes the buffer. This is a no-op if the new size is smaller than
  /// or equal to the current capacity.
  void resize(size_t newSize) {
    mExternal_buffer = nullptr;
    mSize = newSize;
    if (mCapacity < newSize) {
      freeFn(mBuffer);
      if (!allocFn(&mBuffer, this->nbBytes())) {
        throw std::bad_alloc{};
      }
      mCapacity = newSize;
    }
  }

  /// \brief Overload of resize that accepts Dims
  void resize(const nvinfer1::Dims& dims) { return this->resize(volume(dims)); }

  ~GenericBuffer() {
    mExternal_buffer = nullptr;
    freeFn(mBuffer);
  }

  /// \brief Set user memory buffer for TRT Buffer
  void SetExternalData(size_t size, nvinfer1::DataType type, void* buffer) {
    mSize = mCapacity = size;
    mType = type;
    mExternal_buffer = const_cast<void*>(buffer);
  }

  /// \brief Set user memory buffer for TRT Buffer
  void SetExternalData(const nvinfer1::Dims& dims, const void* buffer) {
    mSize = mCapacity = volume(dims);
    mExternal_buffer = const_cast<void*>(buffer);
  }

 private:
  size_t mSize{0}, mCapacity{0};
  nvinfer1::DataType mType;
  void* mBuffer;
  void* mExternal_buffer;
  AllocFunc allocFn;
  FreeFunc freeFn;
};

class DeviceAllocator {
 public:
  bool operator()(void** ptr, size_t size) const {
    return cudaMalloc(ptr, size) == cudaSuccess;
  }
};

class DeviceFree {
 public:
  void operator()(void* ptr) const { cudaFree(ptr); }
};

class HostAllocator {
 public:
  bool operator()(void** ptr, size_t size) const {
    *ptr = malloc(size);
    return *ptr != nullptr;
  }
};

class HostFree {
 public:
  void operator()(void* ptr) const { free(ptr); }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

}  // namespace tensorrt
}  // namespace inference
}  // namespace zetton
