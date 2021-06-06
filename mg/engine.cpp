#include "mg/engine.h"

#include <absl/container/flat_hash_set.h>
#include <fmt/core.h>
#include <glog/logging.h>

namespace mg {

Value operator+(const Value& lhs, const Value& rhs) {
  Value out{lhs.data() + rhs.data()};
  out.grad_->func = [=]() {
    lhs.grad_->data += out.grad();
    rhs.grad_->data += out.grad();
  };
  out.children_ = {lhs, rhs};
  return out;
}

Value operator*(const Value& lhs, const Value& rhs) {
  Value out{lhs.data() * rhs.data()};
  out.grad_->func = [=]() {
    lhs.grad_->data += rhs.data() * out.grad();
    rhs.grad_->data += lhs.data() * out.grad();
  };
  out.children_ = {lhs, rhs};
  return out;
}

void BuildTopoImpl(const Value& v,
                   std::vector<Value>& topo,
                   absl::flat_hash_set<const Value::Data*>& visited) {
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
  absl::flat_hash_set<const Data*> visited;
  BuildTopoImpl(*this, topo, visited);

  std::reverse(topo.begin(), topo.end());

  grad_->data = 1.0;

  for (auto& v : topo) {
    LOG(INFO) << "backward " << v;
    v.grad_->func();
  }
}

std::string Value::Repr() const {
  return fmt::format("Value(data={}, grad={})", data(), grad());
}

std::ostream& operator<<(std::ostream& os, const Value& v) {
  return os << v.Repr();
}

}  // namespace mg