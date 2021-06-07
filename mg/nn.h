#pragma once

#include "mg/value.h"

namespace mg {

class ModuleBase {
 public:
  virtual ~ModuleBase() noexcept = default;

  void ZeroGrad();
  virtual std::vector<Value> Parameters() const { return {}; }
  virtual std::string Repr() const { return "ModuleBase()"; }
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

  void InitWeights();
  Value Forward(const std::vector<Value>& x);

  std::vector<Value> Parameters() const override;
  std::string Repr() const override;

  std::vector<Value> w_;
  Value b_;
  bool nonlin_{true};
};

class Layer final : public ModuleBase {};

class MLP final : public ModuleBase {};

}  // namespace mg