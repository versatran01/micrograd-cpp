#include "mg/engine.h"

#include <fmt/core.h>

namespace mg {

Value& Value::operator+=(const Value& rhs) {
  //
  return *this;
}

Value& Value::operator-=(const Value& rhs) {
  //
  return *this;
}

Value& Value::operator*=(const Value& rhs) {
  //
  return *this;
}

Value& Value::operator/=(const Value& rhs) {
  //
  return *this;
}

Value Value::ReLU() {
  //
  return {};
}

void Value::Backward() {}

std::string Value::Repr() const {
  return fmt::format("Value(data={}, grad={})", data_, grad_);
}

}  // namespace mg