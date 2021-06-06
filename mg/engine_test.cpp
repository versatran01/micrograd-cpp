#include "mg/engine.h"

#include <doctest/doctest.h>

namespace {
using mg::Value;

TEST_CASE("sanity check") {
  auto x = Value(-4.0);
  auto z = 2 * x + 2 + x;
  auto q = z.ReLU() + z * x;
  auto h = (z * z).ReLU();
  auto y = h + q + q * x;
  y.Backward();

  CHECK(y.data() == -20.0);
  CHECK(x.grad() == 46.0);
}

}  // namespace