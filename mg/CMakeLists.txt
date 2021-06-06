cc_library(
  NAME test_main
  SRCS test_main.cpp
  DEPS doctest::doctest)

cc_library(
  NAME engine
  HDRS engine.h
  SRCS engine.cpp
  DEPS mg_base)

cc_test(
  NAME engine_test
  SRCS engine_test.cpp
  DEPS mg_engine mg_test_main OTHER)
