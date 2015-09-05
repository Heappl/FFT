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
            if (i + j < first.size())
                ret[i] += first[i + j] * second[j];
    return ret;
}

TEST(ConvolutionTest, simple_1d_convolution)
{
    for (auto i : {2u, 3u, 8u, 11u, 16u, 32u, 33u, 63u, 128u, 1024u})
    {
        for (auto j : {2u, 2u, 8u, 11u, 12u, 16u, 128u, 256u})
        {
            if (j > i) continue;
            auto vals = generate(i);
            auto filter = generate(j);

            auto expected = naive_convolve(vals, filter);
            auto result = convolution::convolve(vals, filter);

            ASSERT_NO_FATAL_FAILURE(equal(expected, result)) << "sizes: " << i << " and " << j;
        }
    }
}

