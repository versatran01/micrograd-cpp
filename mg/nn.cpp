#include "mg/nn.h"

#include <absl/random/random.h>
#include <fmt/ranges.h>
#include <glog/logging.h>

#include <numeric>  // inner_product

namespace mg {

static absl::BitGen kBitGen;

void ModuleBase::ZeroGrad() {
  for (auto& p : Params()) {
    p.ZeroGrad();
  }
}

Neuron::Neuron(int n_in, bool nonlin) : nonlin_{nonlin} { ws_.resize(n_in); }

void Neuron::UniformInit(double lb, double ub) {
  DataVec ws;
  ws.reserve(ws_.size());
  for (size_t i = 0; i < ws.size(); ++i) {
    ws[i] = absl::Uniform(kBitGen, lb, ub);
  }
  double b = absl::Uniform(kBitGen, lb, ub);
  Init(ws, b);
}

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

Value Neuron::Forward(const ValueVec& x) {
  CHECK_EQ(x.size(), ws_.size());
  auto y = std::inner_product(ws_.begin(), ws_.end(), x.begin(), b_);
  return nonlin_ ? y.ReLU() : y;
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