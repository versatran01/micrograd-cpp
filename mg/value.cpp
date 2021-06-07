#include "mg/value.h"

#include <absl/container/flat_hash_set.h>
#include <fmt/core.h>
#include <glog/logging.h>

namespace mg {

namespace {

// Build a dag, use pointer to (shared) data as identifier
void BuildTopoImpl(const Value& v,
                   std::vector<Value>& topo,
                   absl::flat_hash_set<const Value::DataType*>& visited) {
  if (!visited.contains(v.ptr())) {
    visited.insert(v.ptr());
    for (const auto& child : v.children_) {
      BuildTopoImpl(child, topo, visited);
    }
    topo.push_back(v);
  }
}

}  // namespace

Value operator-(const Value& rhs) { return rhs * -1.0; }

Value operator+(const Value& lhs, const Value& rhs) {
  Value out{lhs.Data() + rhs.Data()};

  // Need to capture by copy
  out.impl_->func = [out, lhs = Value(lhs), rhs = Value(rhs)]() {
    lhs.impl_->grad += out.Grad();
    rhs.impl_->grad += out.Grad();
  };
  out.children_ = {lhs, rhs};
  return out;
}

Value operator-(const Value& lhs, const Value& rhs) { return lhs + (-rhs); }

Value operator*(const Value& lhs, const Value& rhs) {
  Value out{lhs.Data() * rhs.Data()};

  out.impl_->func = [out, lhs = Value(lhs), rhs = Value(rhs)]() {
    lhs.impl_->grad += rhs.Data() * out.Grad();
    rhs.impl_->grad += lhs.Data() * out.Grad();
  };
  out.children_ = {lhs, rhs};
  return out;
}

Value Value::Pow(const DataType& exponent) const {
  Value out{std::pow(Data(), exponent)};

  out.impl_->func = [out, exponent, self = *this]() {
    self.impl_->grad += exponent * out.Data() / self.Data() * out.Grad();
  };
  out.children_ = {*this};
  return out;
}

Value Value::ReLU() const {
  bool gt0 = Data() > 0;
  Value out{gt0 ? Data() : 0};

  out.impl_->func = [=, self = *this]() {
    self.impl_->grad += gt0 * out.Grad();
  };
  out.children_ = {*this};
  return out;
}

Value operator/(const Value& lhs, const Value& rhs) {
  return lhs * rhs.Pow(-1.0);
}

void Value::Backward() {
  std::vector<Value> topo;
  absl::flat_hash_set<const DataType*> visited;
  BuildTopoImpl(*this, topo, visited);

  std::reverse(topo.begin(), topo.end());

  Grad_() = 1.0;

  for (auto& v : topo) {
    v.impl_->func();
  }
}

void Value::ZeroGrad() { impl_->grad = 0.0; }

std::string Value::Repr() const {
  return fmt::format("Value(data={}, grad={})", Data(), Grad());
}

std::ostream& operator<<(std::ostream& os, const Value& v) {
  return os << v.Repr();
}

}  // namespace mg