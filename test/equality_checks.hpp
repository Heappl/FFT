#pragma once

#include <gtest/gtest.h>
#include <vector>
#include <complex>
#include "matrix.hpp"

template <typename IndexableContainer>
inline void equal(const IndexableContainer& expected,
                  const IndexableContainer& result)
{
    ASSERT_EQ(expected.size(), result.size());
    for (auto i = 0u; i < expected.size(); ++i)
        ASSERT_FLOAT_EQ(expected[i], result[i]) << "elem of index: " << i;
}

template <typename T>
inline void equal(const std::vector<std::complex<T>>& expected,
                  const std::vector<std::complex<T>>& result)
{
    ASSERT_EQ(expected.size(), result.size());
    for (auto i = 0u; i < expected.size(); ++i)
        EXPECT_TRUE(abs(expected[i]) - abs(result[i]) < 1.0e-10 * abs(expected[i])) << "elem of index: " << i << " expected: " << expected[i] << " but result is: " << result[i];
}

