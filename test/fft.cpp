#include <gtest/gtest.h>
#include "dft.hpp"
#include "fft.hpp"
#include "generator.hpp"
#include "equality_checks.hpp"

struct FFTTest : ::testing::Test {};

TEST_F(FFTTest, check_fft_single_element)
{
    auto vals = generate(1);
    auto converted = dft::real(fft::fft(vals));
    ASSERT_NO_FATAL_FAILURE(equal(vals, converted));
}

TEST_F(FFTTest, check_fft_two_elements_vs_dft)
{
    auto vals = generate(2);
    auto fft_result = fft::fft(vals);
    auto dft_result = dft::dft(dft::real2complex(vals));
    ASSERT_NO_FATAL_FAILURE(equal(dft_result, fft_result));
}

TEST_F(FFTTest, check_fft_four_elements_vs_dft)
{
    auto vals = generate(4);
    auto fft_result = fft::fft(vals);
    auto dft_result = dft::dft(dft::real2complex(vals));
    ASSERT_NO_FATAL_FAILURE(equal(dft_result, fft_result));
}

TEST_F(FFTTest, check_fft_with_inverse_finishes_the_same)
{
    auto vals = generate(4);
    auto converted = dft::real(fft::inv_fft(fft::fft(vals)));
    ASSERT_NO_FATAL_FAILURE(equal(vals, converted));
}

TEST_F(FFTTest, check_fft_different_sizes_vs_dft)
{
    for (auto i = 8; i <= 512; i *= 2)
    {
        auto vals = generate(i);
        auto fft_result = fft::fft(vals);
        auto dft_result = dft::dft(dft::real2complex(vals));
        ASSERT_NO_FATAL_FAILURE(equal(dft_result, fft_result)) << " for size of " << i;
    }
}

