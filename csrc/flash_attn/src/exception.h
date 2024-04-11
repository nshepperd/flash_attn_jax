#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <stdio.h>

inline void cuda_check_implementation(const cudaError_t err, std::string filename, std::string func, uint32_t line) {
	if (err != cudaSuccess) {
		std::string check_message;
		check_message.append("CUDA error: ");
		check_message.append(cudaGetErrorString(err));
		check_message.append("\n");
		fprintf(stderr, "%s\n", check_message.c_str());
		abort();
	}
}

#define C10_CUDA_CHECK(EXPR)                                        \
  do {                                                              \
    const cudaError_t __err = EXPR;                                 \
    cuda_check_implementation(                                      \
        __err,                                \
        __FILE__,                                                   \
        __func__,                                                   \
        static_cast<uint32_t>(__LINE__));                           \
  } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

inline void c_check_implementation(bool expr, std::string filename, std::string func, uint32_t line) {
	if (!expr) {
		std::string check_message;
		check_message.append("Assert failed at " + filename + " in " + func);
		fprintf(stderr, "%s: line %i\n", check_message.c_str(), line);
		abort();
	}
}

#define C_CHECK(EXPR)                                        \
  do {                                                              \
    const bool __err = EXPR;                                 \
    c_check_implementation(                                      \
        __err,                                \
        __FILE__,                                                   \
        __func__,                                                   \
        static_cast<uint32_t>(__LINE__));                           \
  } while (0)
