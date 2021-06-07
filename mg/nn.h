#pragma once

#include "mg/value.h"

namespace mg {

using ValueVec = std::vector<Value>;
using DataVec = std::vector<Value::DataType>;

class ModuleBase {
 public:
  virtual ~ModuleBase() noexcept = default;

  void ZeroGrad();

  virtual ValueVec Params() const { return {}; }
  virtual DataVec RawParams() const { return {}; }
  virtual std::string Repr() const { return "ModuleBase()"; }

  virtual Value Forward(const ValueVec& x) = 0;
  Value operator()(const ValueVec& x) { return Forward(x); }
};

class Module {
 public:
  Module() = default;

  template <typename T>
  Module(T x) : self_(std::make_shared<T>(std::move(x))) {}

 protected:
  std::shared_ptr<ModuleBase> self_;
};

class Neuron final : public ModuleBase {
 public:
  Neuron(int n_in, bool nonlin = true);

  /// if random = true, random init, otherwise set it to vector
  void UniformInit(double lb = 0.0, double ub = 1.0);
  void ConstInit(double w, double b = 1.0);
  void Init(const std::vector<double>& ws, double b);

  Value Forward(const ValueVec& x) override;

  ValueVec Params() const override;
  DataVec RawParams() const override;
  std::string Repr() const override;

  ValueVec ws_;
  Value b_;
  bool nonlin_{true};
};

class Layer final : public ModuleBase {};

class MLP final : public ModuleBase {};

}  // namespace mg