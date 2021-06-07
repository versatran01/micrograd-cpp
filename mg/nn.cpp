#include "mg/nn.h"

#include <fmt/ranges.h>
#include <glog/logging.h>

namespace mg {

void ModuleBase::ZeroGrad() {
  for (auto& p : Params()) {
    p.ZeroGrad();
  }
}

Neuron::Neuron(int n_in, bool nonlin) : nonlin_{nonlin} { ws_.resize(n_in); }

void Neuron::ConstInit(double w, double b) {
  auto ws = std::vector<double>(ws_.size(), w);
  Init(ws, b);
}

void Neuron::Init(const std::vector<double>& ws, double b) {
  CHECK_EQ(ws.size(), ws_.size());
  for (size_t i = 0; i < ws_.size(); ++i) {
    ws_[i].Data_() = ws[i];
  }
  b_.Data_() = b;
}

ValueVec Neuron::Forward(const ValueVec& x) {
  //
  return x;
}

ValueVec Neuron::Params() const {
  ValueVec params = ws_;
  params.push_back(b_);
  return params;
}

DataVec Neuron::RawParams() const {
  DataVec dv;
  for (const auto& w : ws_) {
    dv.push_back(w.Data());
  }
  dv.push_back(b_.Data());
  return dv;
}

std::string Neuron::Repr() const {
  return fmt::format("Neuron(n_in={}, nonlin={})", ws_.size(), nonlin_);
}

}  // namespace mg