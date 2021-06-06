#include "mg/engine.h"

#include <doctest/doctest.h>

namespace {
using mg::Value;

TEST_CASE("construction") {
  Value x;
  CHECK(x.data() == 0.0);
  Value y{2.0};
  CHECK(y.data() == 2.0);

  Value z = y;
  CHECK(z.data() == 2.0);
  CHECK(y == z);
  CHECK(Eqq(y, z));
}

TEST_CASE("operator") {
  auto x = Value(2.0);
  auto y = Value(3.0);
  REQUIRE(x.data() == 2.0);
  REQUIRE(y.data() == 3.0);

  SUBCASE("add") {
    auto z = x + y;

    CHECK(z.data() == 5.0);
  }

  SUBCASE("mul") {
    auto z = x * y;

    CHECK(z.data() == 6.0);
  }
}

TEST_CASE("backward") {
  auto x = Value(2.0);
  auto y = Value(3.0);
  auto z = x + y;
  z.Backward();

  CHECK(x.grad() == 1.0);
  CHECK(y.grad() == 1.0);
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