#pragma once

#include <af/array.h>

#include <functional>
#include <vector>

namespace mg {

/// Stores a single scalar value and its gradient
class Value {
 public:
  using GradFunc = std::function<void()>;

  Value(double d);
  Value(const af::array& data);

  Value ReLU();

  af::array data_;
};


}  // namespace mg