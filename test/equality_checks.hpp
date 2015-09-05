#pragma once

#include <gtest/gtest.h>
#include <vector>

template <typename IndexableContainer>
inline void equal(const IndexableContainer& expected,
                  const IndexableContainer& result)
{
    ASSERT_EQ(expected.size(), result.size());
    for (auto i = 0u; i < expected.size(); ++i)
        ASSERT_FLOAT_EQ(expected[i], result[i]) << "elem of index: " << i;
}

