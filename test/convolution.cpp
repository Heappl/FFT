#include <gtest/gtest.h>
#include "convolution.hpp"
#include "generator.hpp"
#include "equality_checks.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>

template <typename T>
std::vector<T> naive_convolve(std::vector<T> first, std::vector<T> second)
{
    std::vector<T> ret(first.size(), 0);
    for (auto i = 0u; i < first.size(); ++i)
        for (auto j = 0u; j < second.size(); ++j)
            ret[i] = first[i + j] * second[j];
    return ret;
}

TEST(ConvolutionTest, simple_1d_convolution)
{
    auto vals = generate(100);
    auto filter = generate(10);

    auto expected = naive_convolve(vals, filter);
    auto result = convolution::convolve(vals, filter);

    ASSERT_NO_FATAL_FAILURE(equal(expected, result));
}

