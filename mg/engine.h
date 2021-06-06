#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace mg {

/// Stores a single scalar value and its gradient
class Value {
 public:
  using Data = double;
  using GradFunc = std::function<void()>;

  // Constructors
  Value() = default;
  explicit Value(Data d) : data_{std::make_shared<Data>(d)} {}

  // Accesser
  Data data() const { return *data_; }
  Data grad() const { return grad_->data; }
  const Data* ptr() const { return data_.get(); }

  // Operators
  Value operator-();

  // Value& operator+=(const Value& rhs);
  // Value& operator-=(const Value& rhs);
  // Value& operator*=(const Value& rhs);
  // Value& operator/=(const Value& rhs);

  friend Value operator+(const Value& lhs, const Value& rhs);
  friend Value operator-(const Value& lhs, const Value& rhs);
  friend Value operator*(const Value& lhs, const Value& rhs);
  friend Value operator/(const Value& lhs, const Value& rhs);

  friend bool operator==(const Value& lhs, const Value& rhs) {
    return lhs.data() == rhs.data();
  }
  friend bool operator!=(const Value& lhs, const Value& rhs) {
    return !(lhs == rhs);
  }

  friend bool Eqq(const Value& lhs, const Value& rhs) {
    return lhs.ptr() == rhs.ptr();
  }

  Value ReLU() { return {}; }

  void Backward();

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const Value& v);

  using SharedData = std::shared_ptr<Data>;
  struct Grad {
    Data data{};
    GradFunc func{[]() {}};
  };
  using SharedGrad = std::shared_ptr<Grad>;

  SharedData data_{std::make_shared<Data>()};
  SharedGrad grad_{std::make_shared<Grad>()};
  std::vector<Value> children_;
};

}  // namespace mg