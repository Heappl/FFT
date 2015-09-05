#include <gtest/gtest.h>
#include "dft.hpp"
#include "generator.hpp"
#include "equality_checks.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

struct DftTest : ::testing::Test {};

TEST_F(DftTest, check_dft_with_inverse_finishes_the_same)
{
    auto vals = generate();
    auto converted = dft::real(dft::inv_dft(dft::dft(vals)));
    ASSERT_NO_FATAL_FAILURE(equal(vals, converted));
}

TEST_F(DftTest, check_dft_parsevals_property)
{
    auto x = generate();
    auto y = generate();

    auto X = dft::dft(x);
    auto Y = dft::dft(y);

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

    auto X = dft::dft(x);
    auto Y = dft::dft(y);

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

