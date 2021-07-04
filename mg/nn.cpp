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

// Neuron
Neuron::Neuron(size_t n_in, bool nonlin) : nonlin_{nonlin} { ws_.resize(n_in); }

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

ValueVec Neuron::Forward(const ValueVec& x) {
  CHECK_EQ(x.size(), ws_.size());
  auto y = std::inner_product(ws_.begin(), ws_.end(), x.begin(), b_);
  auto z = nonlin_ ? y.ReLU() : y;
  return {z};
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

// Layer
Layer::Layer(size_t n_in, size_t n_out, bool nonlin)
    : n_in{n_in}, n_out{n_out} {
  neurons.reserve(n_out);
  for (size_t i = 0; i < n_out; ++i) {
    neurons.push_back(Neuron(n_in, nonlin));
  }
}

ValueVec Layer::Forward(const ValueVec& x) {
  CHECK_EQ(x.size(), n_in);
  ValueVec v;
  v.reserve(n_out);
  for (auto& n : neurons) {
    v.push_back(n(x)[0]);
  }
  return v;
}

ValueVec Layer::Params() const {
  ValueVec v;
  v.reserve(n_in * n_out);
  for (const auto& n : neurons) {
    auto ws = n.Params();
    v.insert(v.end(), ws.cbegin(), ws.cend());
  }
  return v;
}

DataVec Layer::RawParams() const {
  DataVec v;
  v.reserve(n_in * n_out);
  for (const auto& n : neurons) {
    auto ws = n.RawParams();
    v.insert(v.end(), ws.cbegin(), ws.cend());
  }
  return v;
}

std::string Layer::Repr() const {
  return fmt::format("Layer(n_in={}, n_out={})", n_in, n_out);
}

// MLP
MLP::MLP(size_t n_in, const std::vector<size_t>& n_out)
    : n_in{n_in}, n_out{n_out} {
  for (size_t i = 0; i < n_out.size(); ++i) {
    layers.push_back(
        Layer{n_in, n_out[i], i == n_out.size() - 1 ? false : true});
    n_in = n_out[i];
  }
}

ValueVec MLP::Forward(const ValueVec& x) {
  ValueVec z = x;  // make a copy
  for (auto& l : layers) {
    z = l(z);
  }
  return z;
}

ValueVec MLP::Params() const {
  ValueVec v;
  for (const auto& l : layers) {
    auto ws = l.Params();
    v.insert(v.end(), ws.cbegin(), ws.cend());
  }
  return v;
}

DataVec MLP::RawParams() const {
  DataVec v;
  for (const auto& l : layers) {
    auto ws = l.RawParams();
    v.insert(v.end(), ws.cbegin(), ws.cend());
  }
  return v;
}

std::string MLP::Repr() const {
  return fmt::format("MLP(n_in={}, n_out={}", n_in, n_out);
}

}  // namespace mg
