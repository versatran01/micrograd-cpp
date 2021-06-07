#include "mg/value.h"

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
  auto x = Value{2.0};
  auto y = Value{-3.0};

  SUBCASE("scope") {
    Value z;
    {
      auto w = Value{4.0};
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

  SUBCASE("pow") {
    auto z = x.Pow(3);
    z.Backward();

    CHECK(x.Grad() == 12.0);
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
  auto x = Value{-4.0};
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

TEST_CASE("More ops") {
  auto a = Value{-4.0f};
  auto b = Value{2.0f};

  SUBCASE("case 1") {
    auto c = a + b;
    auto d = a * b + b * b * b;
    c = c + c + 1.0;
    c = c + 1.0 + c + (-a);
    d = d + d * 2.0 + (b + a).ReLU();
    d.Backward();
    CHECK(d.Data() == 0.0);
    CHECK(a.Grad() == 6.0);
    CHECK(b.Grad() == 24.0);
  }

  SUBCASE("case 2") {
    auto c = a + b;
    auto d = a * b + b * b * b;
    c = c + c + 1.0;
    c = c + 1.0 + c + (-a);
    d = d + d * 2.0 + (b + a).ReLU();
    d = d + 3.0 * d + (b - a).ReLU();
    d.Backward();
    CHECK(d.Data() == 6.0);
    CHECK(a.Grad() == 23.0);
    CHECK(b.Grad() == 97.0);
  }

  SUBCASE("case 3") {
    auto c = a + b;
    auto d = a * b + b * b * b;
    c = c + c + 1.0;
    c = c + 1.0 + c + (-a);
    d = d + d * 2.0 + (b + a).ReLU();
    d = d + 3.0 * d + (b - a).ReLU();
    auto e = c - d;
    auto f = e * e;
    auto g = f / 2.0;
    g = g + 10.0 / f;
    g.Backward();
    CHECK(g.Data() == doctest::Approx(24.704082));
    CHECK(a.Grad() == doctest::Approx(138.833819));
    CHECK(b.Grad() == doctest::Approx(645.577259));
  }
}

}  // namespace