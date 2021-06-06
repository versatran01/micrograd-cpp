#include "mg/engine.h"

#include <doctest/doctest.h>

namespace {
using mg::Value;

TEST_CASE("add value") {
  auto x = Value(2.0);
  auto y = Value(3.0);
  auto z = x + y;

  CHECK(x.data() == 2.0);
  CHECK(y.data() == 3.0);
  CHECK(z.data() == 5.0);
}

// TEST_CASE("sanity check") {
//   auto x = Value(-4.0);
//   auto z = 2 * x + 2 + x;
//   auto q = z.ReLU() + z * x;
//   auto h = (z * z).ReLU();
//   auto y = h + q + q * x;
//   y.Backward();

//   CHECK(y.data() == -20.0);
//   CHECK(x.grad() == 46.0);
// }

}  // namespace