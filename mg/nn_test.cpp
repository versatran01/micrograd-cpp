#include "mg/nn.h"

#include <doctest/doctest.h>
#include <fmt/ranges.h>

#include <sstream>

namespace doctest {
template <typename T>
struct StringMaker<std::vector<T>> {
  static String convert(const std::vector<T>& in) {
    std::ostringstream oss;
    oss << fmt::format("{}", in);
    return oss.str().c_str();
  }
};
}  // namespace doctest

namespace {

using namespace mg;

TEST_CASE("Neuron") {
  auto n = Neuron{3, true};

  SUBCASE("Init") {
    n.Init({2, 3, 4}, 5);
    CHECK(n.RawParams() == DataVec{2, 3, 4, 5});
  }

  SUBCASE("ConstInit") {
    n.ConstInit(2, 5);
    CHECK(n.RawParams() == DataVec{2, 2, 2, 5});
  }

  SUBCASE("Forward Positive") {
    n.ConstInit(1, 1);
    ValueVec x = {1, 2, 3};
    auto y = n.Forward(x);
    REQUIRE(y.size() == 1);
    CHECK(y[0].Data() == 7);
  }

  SUBCASE("Forward Negative") {
    n.ConstInit(1, 1);
    ValueVec x = {-1, -2, -3};
    auto y = n.Forward(x);
    REQUIRE(y.size() == 1);
    CHECK(y[0].Data() == 0);
  }
}

}  // namespace