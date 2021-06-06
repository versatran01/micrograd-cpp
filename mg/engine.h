#pragma once

#include <functional>
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
  Value(Data d) : data_{d} {}

  // Accesser
  Data data() const { return data_; }
  Data grad() const { return grad_; }

  // Operators
  Value operator-();

  Value& operator+=(const Value& rhs);
  Value& operator-=(const Value& rhs);
  Value& operator*=(const Value& rhs);
  Value& operator/=(const Value& rhs);

  friend Value operator+(Value lhs, const Value& rhs) { return lhs += rhs; }
  friend Value operator-(Value lhs, const Value& rhs) { return lhs -= rhs; }
  friend Value operator*(Value lhs, const Value& rhs) { return lhs *= rhs; }
  friend Value operator/(Value lhs, const Value& rhs) { return lhs /= rhs; }

  friend bool operator==(const Value& lhs, const Value& rhs) {
    return lhs.data() == rhs.data();
  }
  friend bool operator!=(const Value& lhs, const Value& rhs) {
    return !(lhs == rhs);
  }

  Value ReLU();

  void Backward();

  std::string Repr() const;

 private:
  Data data_{};
  Data grad_{};
  GradFunc backward_{nullptr};
};

}  // namespace mg