#include "mg/nn.h"

#include <fmt/core.h>

namespace mg {

void ModuleBase::ZeroGrad() {
  for (auto& p : Parameters()) {
    p.ZeroGrad();
  }
}

std::vector<Value> Neuron::Parameters() const {
  auto params = w_;
  params.push_back(b_);
  return params;
}

std::string Neuron::Repr() const {
  return fmt::format("Neuron(n_in={}, nonline={})", w_.size(), nonlin_);
}

}  // namespace mg