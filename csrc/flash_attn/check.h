#pragma once

#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <string>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

class CheckHelper {
public:
  explicit CheckHelper(std::string expr) : expr_(expr) {}

  template <typename T> inline CheckHelper &operator<<(const T &value) {
    stream_ << value;
    return *this;
  }

  inline CheckHelper &operator<<(ffi::ErrorCode errc) {
    errc_ = errc;
    return *this;
  }

  inline operator ffi::Error() {
    std::ostringstream full_message;
    full_message << "Check failed: " << expr_;
    std::string additional = stream_.str();
    if (!additional.empty()) {
      full_message << "; " << additional;
    }
    return ffi::Error(errc_, full_message.str());
  }

private:
  ffi::ErrorCode errc_ = ffi::ErrorCode::kUnknown;
  std::string expr_;
  std::ostringstream stream_;
};

#define FFI_CHECK(expr)                                                                            \
  static_assert(!std::is_same_v<decltype(expr), cudaError_t>,                                      \
                "Use FFI_CUDA_CHECK for CUDA error codes, not FFI_CHECK.");                        \
  if (!(expr))                                                                                     \
  return CheckHelper(#expr)

#define FFI_CUDA_CHECK(expr)                                                                       \
  static_assert(std::is_same_v<decltype(expr), cudaError_t>,                                       \
                "Expect cudaError_t for FFI_CUDA_CHECK.");                                         \
  if (cudaError_t _cuda_check = (expr); _cuda_check != cudaSuccess)                                \
  return CheckHelper(std::string(#expr)) << " CUDA Error: " << cudaGetErrorString(_cuda_check)

#define FFI_CHECK_OPTIONAL(dest, expr)                                                             \
  if (auto _opt = (expr); _opt.has_value())                                                        \
    dest = _opt.value();                                                                           \
  else                                                                                             \
    return CheckHelper(std::string(#expr))

#define FFI_RET_CHECK(expr)                                                                        \
  if (auto _error = (expr); !_error.success())                                                     \
  return _error

#define FFI_CHECK_ALLOC(dest, expr)                                                                \
  void *dest = nullptr;                                                                            \
  if (auto _opt = (expr); _opt.has_value())                                                        \
    dest = _opt.value();                                                                           \
  else                                                                                             \
    return CheckHelper(std::string(#expr))
