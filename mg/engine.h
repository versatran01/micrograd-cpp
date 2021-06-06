#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mg {

static auto kNoop = []() {};

/// Stores a single scalar value and its gradient
class Value {
 public:
  using DataType = double;
  using GradFunc = std::function<void()>;

  // Constructors
  Value() = default;
  Value(DataType d) { impl_->data = d; }

  // Accesser
  DataType Data() const { return impl_->data; }
  DataType Grad() const { return impl_->grad; }
  DataType& Data_() { return impl_->data; }
  DataType& Grad_() { return impl_->grad; }
  const DataType* ptr() const { return &impl_->data; }

  // Operators
  friend Value operator-(const Value& rhs);
  friend Value operator+(const Value& lhs, const Value& rhs);
  friend Value operator-(const Value& lhs, const Value& rhs);
  friend Value operator*(const Value& lhs, const Value& rhs);
  friend Value operator/(const Value& lhs, const Value& rhs);

  friend bool operator==(const Value& lhs, const Value& rhs) {
    return lhs.Data() == rhs.Data();
  }
  friend bool operator!=(const Value& lhs, const Value& rhs) {
    return !(lhs == rhs);
  }
  friend bool Eqq(const Value& lhs, const Value& rhs) {
    return lhs.ptr() == rhs.ptr();
  }

  Value Pow(const DataType& exponent) const;
  Value ReLU() const;

  void Backward();

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const Value& v);

  struct ValueImpl {
    DataType data{};
    DataType grad{};
    GradFunc func{kNoop};
  };

  std::shared_ptr<ValueImpl> impl_{std::make_shared<ValueImpl>()};
  std::vector<Value> children_;
};

}  // namespace mg