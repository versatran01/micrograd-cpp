#include "mg/engine.h"

#include <absl/container/flat_hash_set.h>
#include <fmt/core.h>
#include <glog/logging.h>

namespace mg {

Value operator+(const Value& lhs, const Value& rhs) {
  Value out{lhs.Data() + rhs.Data()};
  out.impl_->func = [=]() {
    lhs.impl_->grad += out.Grad();
    rhs.impl_->grad += out.Grad();
  };
  out.children_ = {lhs, rhs};
  return out;
}

Value operator*(const Value& lhs, const Value& rhs) {
  Value out{lhs.Data() * rhs.Data()};
  out.impl_->func = [=]() {
    lhs.impl_->grad += rhs.Data() * out.Grad();
    rhs.impl_->grad += lhs.Data() * out.Grad();
  };
  out.children_ = {lhs, rhs};
  return out;
}

void BuildTopoImpl(const Value& v,
                   std::vector<Value>& topo,
                   absl::flat_hash_set<const Value::DataType*>& visited) {
  if (!visited.contains(v.ptr())) {
    visited.insert(v.ptr());
    for (const auto& child : v.children_) {
      BuildTopoImpl(child, topo, visited);
    }
    LOG(INFO) << "add " << v;
    topo.push_back(v);
  }
}

void Value::Backward() {
  std::vector<Value> topo;
  absl::flat_hash_set<const DataType*> visited;
  BuildTopoImpl(*this, topo, visited);

  std::reverse(topo.begin(), topo.end());

  Grad_() = 1.0;

  for (auto& v : topo) {
    LOG(INFO) << "backward " << v;
    v.impl_->func();
  }
}

std::string Value::Repr() const {
  return fmt::format("Value(data={}, grad={})", Data(), Grad());
}

std::ostream& operator<<(std::ostream& os, const Value& v) {
  return os << v.Repr();
}

}  // namespace mg