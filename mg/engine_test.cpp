#include "mg/engine.h"

#include <doctest/doctest.h>

namespace {
using mg::Value;

TEST_CASE("construction") {
  Value x;
  CHECK(x.Data() == 0.0);
  Value y{2.0};
  CHECK(y.Data() == 2.0);

  Value z = y;
  CHECK(z.Data() == 2.0);
  CHECK(y == z);
  CHECK(Eqq(y, z));
}

TEST_CASE("operator") {
  auto x = Value(2.0);
  auto y = Value(-3.0);
  REQUIRE(x.Data() == 2.0);
  REQUIRE(y.Data() == -3.0);

  SUBCASE("add") {
    auto z = x + y;
    CHECK(z.Data() == -1.0);
  }

  SUBCASE("mul") {
    auto z = x * y;
    CHECK(z.Data() == -6.0);
  }

  SUBCASE("pow") {
    auto z = x.Pow(2.0);
    CHECK(z.Data() == 4.0);
  }

  SUBCASE("neg") {
    auto z = -x;
    CHECK(z.Data() == -2.0);
  }

  SUBCASE("div") {
    auto z = y / x;
    CHECK(z.Data() == -1.5);
  }

  SUBCASE("relu") {
    CHECK(x.ReLU().Data() == 2.0);
    CHECK(y.ReLU().Data() == 0.0);
  }
}

TEST_CASE("backward") {
  auto x = Value(2.0);
  auto y = Value(-3.0);

  SUBCASE("scope") {
    Value z;
    {
      auto w = Value(4.0);
      z = x * w;
    }
    z.Backward();
    CHECK(z.Data() == 8.0);
    CHECK(x.Grad() == 4.0);
  }

  SUBCASE("add") {
    auto z = x + y;
    z.Backward();

    CHECK(x.Grad() == 1.0);
    CHECK(y.Grad() == 1.0);
  }

  SUBCASE("mul") {
    auto z = x * y;
    z.Backward();
    CHECK(x.Grad() == -3.0);
    CHECK(y.Grad() == 2.0);
  }

  SUBCASE("relu") {
    auto xr = x.ReLU();
    auto yr = y.ReLU();
    xr.Backward();
    yr.Backward();
    CHECK(x.Grad() == 1.0);
    CHECK(y.Grad() == 0.0);
  }
}

TEST_CASE("sanity check") {
  auto x = Value(-4.0);
  auto z = 2.0 * x + 2.0 + x;
  auto q = z.ReLU() + z * x;
  auto h = (z * z).ReLU();
  auto y = h + q + q * x;

  SUBCASE("z") {
    z.Backward();
    CHECK(z.Data() == -10);
    CHECK(x.Grad() == 3);
  }

  SUBCASE("q") {
    q.Backward();
    CHECK(q.Data() == 40);
    CHECK(z.Grad() == -4);
    CHECK(x.Grad() == -22);
  }

  SUBCASE("h") {
    h.Backward();
    CHECK(h.Data() == 100);
    CHECK(z.Grad() == -20);
    CHECK(x.Grad() == -60);
  }

  SUBCASE("y") {
    y.Backward();
    CHECK(y.Data() == -20);
    CHECK(x.Grad() == 46);
  }
}

}  // namespace