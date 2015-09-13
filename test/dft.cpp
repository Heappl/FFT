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
    for (auto i = 2; i < y.size(); ++i)
        y[i] = 0;

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

TEST_F(DftTest, check_2d_dft_with_inverse_finishes_the_same_for_square_matrix)
{
    auto size = 6u;
    auto matrix = generate(size * size);
    auto freq_form = dft::dft_2d(dft::real2complex(matrix), size);
    auto inverted = dft::inv_dft_2d(freq_form, size);
    auto real_of_inverted = dft::real(inverted);

    ASSERT_NO_FATAL_FAILURE(equal(matrix, real_of_inverted));
}

