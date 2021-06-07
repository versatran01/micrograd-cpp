cc_library(
  NAME test_main
  SRCS test_main.cpp
  DEPS doctest::doctest)

cc_library(
  NAME value
  HDRS value.h
  SRCS value.cpp
  DEPS mg_base absl::flat_hash_set)

cc_test(
  NAME value_test
  SRCS value_test.cpp
  DEPS mg_value mg_test_main
  OTHER)

cc_library(
  NAME nn
  HDRS nn.h
  SRCS nn.cpp
  DEPS mg_value)
