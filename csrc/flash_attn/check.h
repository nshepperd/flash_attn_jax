#pragma once

#include <stdio.h>

inline void check_implementation(bool expr, std::string check_message) {
	if (!expr) {
		fprintf(stderr, "%s\n", check_message.c_str());
		abort();
	}
}

#define CHECK(EXPR, MESSAGE)												\
  do {                                                              \
    const bool __err = EXPR;                                 \
    check_implementation(                                      \
        __err,                                \
		MESSAGE);							  \
  } while (0)
