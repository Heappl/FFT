#include <gtest/gtest.h>
#include "dft.hpp"
#include "fft.hpp"
#include "generator.hpp"
#include "equality_checks.hpp"

struct FFTTest : ::testing::Test {};

TEST_F(FFTTest, check_fft_single_element)
{
    auto vals = generate(1);
    auto converted = dft::real(fft::fft(dft::real2complex(vals)));
    ASSERT_NO_FATAL_FAILURE(equal(vals, converted));
}

TEST_F(FFTTest, check_fft_two_elements_vs_dft)
{
    auto vals = generate(2);
    auto fft_result = fft::fft(dft::real2complex(vals));
    auto dft_result = dft::dft(dft::real2complex(vals));
    ASSERT_NO_FATAL_FAILURE(equal(dft_result, fft_result));
}

TEST_F(FFTTest, check_fft_four_elements_vs_dft)
{
    auto vals = generate(4);
    auto fft_result = fft::fft(dft::real2complex(vals));
    auto dft_result = dft::dft(dft::real2complex(vals));
    ASSERT_NO_FATAL_FAILURE(equal(dft_result, fft_result));
}

TEST_F(FFTTest, check_fft_with_inverse_finishes_the_same)
{
    auto vals = generate(4);
    auto converted = dft::real(fft::inv_fft(fft::fft(dft::real2complex(vals))));
    ASSERT_NO_FATAL_FAILURE(equal(vals, converted));
}

TEST_F(FFTTest, check_fft_different_sizes_vs_dft)
{
    for (auto i = 8; i <= 512; i *= 2)
    {
        auto vals = generate(i);
        auto fft_result = fft::fft(dft::real2complex(vals));
        auto dft_result = dft::dft(dft::real2complex(vals));
        ASSERT_NO_FATAL_FAILURE(equal(dft_result, fft_result)) << " for size of " << i;
    }
}

TEST_F(FFTTest, check_fft_2d_single_element)
{
    auto vals = generate(1);
    auto converted = dft::real(fft::fft_2d(dft::real2complex(vals), 1u));
    ASSERT_NO_FATAL_FAILURE(equal(vals, converted));
}

TEST_F(FFTTest, check_fft_2d_vs_dft_square_matrix)
{
    for (auto size : {2u, 4u, 8u, 16u, 32u, 64u})
    {
        auto vals = generate(size * size);
        auto fft_result = fft::fft_2d(dft::real2complex(vals), size);
        auto dft_result = dft::dft_2d(dft::real2complex(vals), size);
        ASSERT_NO_FATAL_FAILURE(equal(dft_result, fft_result)) << "size " << size;
    }
}

TEST_F(FFTTest, check_fft_2d_vs_dft_rectangle_matrix)
{
    for (auto width : {2u, 4u, 8u, 16u, 32u, 64u})
    {
        for (auto height : {2u, 4u, 8u, 16u, 32u, 64u})
        {
            if (width == height) continue;
            auto vals = generate(width * height);
            auto fft_result = fft::fft_2d(dft::real2complex(vals), width);
            auto dft_result = dft::dft_2d(dft::real2complex(vals), width);
            ASSERT_NO_FATAL_FAILURE(equal(dft_result, fft_result))
                << "width " << width << " and height " << height;
        }
    }
}

