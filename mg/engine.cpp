#include "mg/engine.h"

#include <fmt/core.h>

namespace mg {

Value operator+(const Value& lhs, const Value& rhs) {
  Value out{lhs.data() + rhs.data()};
  out.backward_ = [out, lhs, rhs]() {
    *lhs.grad_ += *out.grad_;
    *rhs.grad_ += *out.grad_;
  };
  return out;
}

void Value::Backward() {}

Value ReLU() { return {}; }

std::string Value::Repr() const {
  return fmt::format("Value(data={}, grad={})", data(), grad());
}

}  // namespace mg