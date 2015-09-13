#include <gtest/gtest.h>
#include "dft.hpp"
#include "generator.hpp"
#include "equality_checks.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

struct DftTest : ::testing::Test {};

TEST_F(DftTest, check_dft_single_element)
{
    auto vals = generate(1);
    auto converted = dft::real(dft::dft_2d(dft::real2complex(vals), 1));
    ASSERT_NO_FATAL_FAILURE(equal(vals, converted));
}

TEST_F(DftTest, check_dft_with_inverse_finishes_the_same)
{
    auto vals = generate();
    auto converted = dft::real(dft::inv_dft(dft::dft(dft::real2complex(vals))));
    ASSERT_NO_FATAL_FAILURE(equal(vals, converted));
}

TEST_F(DftTest, check_dft_parsevals_property)
{
    auto x = generate();
    auto y = generate();

    auto X = dft::dft(dft::real2complex(x));
    auto Y = dft::dft(dft::real2complex(y));

    std::complex<double> time_domain_sum;
    for (auto i = 0u; i < x.size(); ++i)
        time_domain_sum += x[i] * y[i];

    std::complex<double> freq_domain_sum;
    for (auto i = 0u; i < X.size(); ++i)
        freq_domain_sum += X[i] * std::conj(Y[i]);
    freq_domain_sum /= double(X.size());

    ASSERT_TRUE(abs(time_domain_sum - freq_domain_sum) < 1.0e-13 * abs(time_domain_sum));
}

TEST_F(DftTest, check_convolution)
{
    auto N = 100;
    auto x = generate(N);
    auto y = generate(N);

    auto X = dft::dft(dft::real2complex(x));
    auto Y = dft::dft(dft::real2complex(y));

    auto freq_mult = decltype(X)(N, 0);
    for (auto i = 0u; i < N; ++i)
        freq_mult[i] = X[i] * Y[i];

    auto inv = dft::real(dft::inv_dft(freq_mult));

    //convolution
    auto expected = decltype(x)(N, 0);
    for (auto n = 0u; n < N; ++n)
        for (auto l = 0u; l < N; ++l)
            expected[n] += x[l] * y[(n + N - l) % N];

    for (auto i = 0u; i < x.size(); ++i)
        ASSERT_FLOAT_EQ(expected[i], inv[i]) << i;
}

TEST_F(DftTest, check_2d_dft_single_element)
{
    auto matrix = generate(1);
    auto converted = dft::real(dft::dft_2d(dft::real2complex(matrix), 1));
    ASSERT_NO_FATAL_FAILURE(equal(matrix, converted));
}

void check_2d_dft_inverse_finishes_the_same(size_t width, size_t height)
{
    auto matrix = generate(width * height);
    auto freq_form = dft::dft_2d(dft::real2complex(matrix), width);
    auto inverted = dft::inv_dft_2d(freq_form, width);
    auto real_of_inverted = dft::real(inverted);

    ASSERT_NO_FATAL_FAILURE(equal(matrix, real_of_inverted));
}

TEST_F(DftTest, check_2d_dft_with_inverse_finishes_the_same_for_square_matrix)
{
    auto size = 6u;
    check_2d_dft_inverse_finishes_the_same(size, size);
}

TEST_F(DftTest, check_2d_dft_with_inverse_finishes_the_same_for_rectangle_matrix)
{
    ASSERT_NO_FATAL_FAILURE(check_2d_dft_inverse_finishes_the_same(7, 11));
    ASSERT_NO_FATAL_FAILURE(check_2d_dft_inverse_finishes_the_same(13, 5));
}

void check_dft_2d_parsevals_property(size_t width, size_t height)
{
    auto size = width * height;
    auto x = generate(size);
    auto y = generate(size);

    auto X = dft::dft_2d(dft::real2complex(x), width);
    auto Y = dft::dft_2d(dft::real2complex(y), width);

    std::complex<double> time_domain_sum;
    for (auto i = 0u; i < x.size(); ++i)
        time_domain_sum += x[i] * y[i];

    std::complex<double> freq_domain_sum;
    for (auto i = 0u; i < X.size(); ++i)
        freq_domain_sum += X[i] * std::conj(Y[i]);
    freq_domain_sum /= double(X.size());

    ASSERT_TRUE(abs(time_domain_sum - freq_domain_sum) < 1.0e-13 * abs(time_domain_sum));
}

TEST_F(DftTest, check_dft_2d_parsevals_property)
{
    ASSERT_NO_FATAL_FAILURE(check_dft_2d_parsevals_property(1, 1));
    ASSERT_NO_FATAL_FAILURE(check_dft_2d_parsevals_property(4, 4));
    ASSERT_NO_FATAL_FAILURE(check_dft_2d_parsevals_property(7, 5));
    ASSERT_NO_FATAL_FAILURE(check_dft_2d_parsevals_property(17, 11));
    ASSERT_NO_FATAL_FAILURE(check_dft_2d_parsevals_property(3, 19));
}

void check_convolution_2d(size_t width, size_t height)
{
    auto N = width * height;
    auto x = generate(N);
    auto y = generate(N);

    auto X = dft::dft_2d(dft::real2complex(x), width);
    auto Y = dft::dft_2d(dft::real2complex(y), width);

    auto freq_mult = decltype(X)(N, 0);
    for (auto i = 0u; i < N; ++i)
        freq_mult[i] = X[i] * Y[i];

    auto inv = dft::real(dft::inv_dft_2d(freq_mult, width));

    //convolution
    auto expected = decltype(x)(N, 0);
    for (auto dest_row = 0u; dest_row < height; ++dest_row)
    {
        for (auto dest_col = 0u; dest_col < width; ++dest_col)
        {
            for (auto src_row = 0u; src_row < height; ++src_row)
            {
                for (auto src_col = 0u; src_col < width; ++src_col)
                {
                    auto y_src_row = (dest_row + height - src_row) % height;
                    auto y_src_col = (dest_col + width - src_col) % width;
                    auto dest_index = dest_row * width + dest_col;
                    auto x_index = src_row * width + src_col;
                    auto y_index = y_src_row * width + y_src_col;
                    expected[dest_index] += x[x_index] * y[y_index];
                }
            }
        }
    }

    for (auto i = 0u; i < x.size(); ++i)
        EXPECT_FLOAT_EQ(expected[i], inv[i]) << i;
}

TEST_F(DftTest, check_convolution_2d)
{
    ASSERT_NO_FATAL_FAILURE(check_convolution_2d(1, 5));
    ASSERT_NO_FATAL_FAILURE(check_convolution_2d(7, 1));
    ASSERT_NO_FATAL_FAILURE(check_convolution_2d(4, 4));
    ASSERT_NO_FATAL_FAILURE(check_convolution_2d(8, 4));
    ASSERT_NO_FATAL_FAILURE(check_convolution_2d(6, 8));
}

