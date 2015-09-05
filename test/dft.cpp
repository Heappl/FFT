#include <gtest/gtest.h>
#include "dft.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>

struct DftTest : ::testing::Test
{
    std::mt19937 rd;
    std::uniform_real_distribution<double> dist;

    DftTest() : rd(), dist(-100, 100) {}

    std::vector<double> generate(size_t size = 100)
    {
        std::vector<double> ret(size);
        for (auto& elem : ret) elem = dist(rd);
        return ret;
    }

    void equal(std::vector<double> expected,
               std::vector<double> result)
    {
        ASSERT_EQ(expected.size(), result.size());
        for (auto i = 0u; i < expected.size(); ++i)
            ASSERT_FLOAT_EQ(expected[i], result[i]) << "elem of index: " << i;
    }
};

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

    ASSERT_TRUE(abs(time_domain_sum - freq_domain_sum) < 1.0e-14 * abs(time_domain_sum));
}

TEST_F(DftTest, check_convolution)
{
    std::vector<double> vals1(100);
    std::vector<double> vals2(100);
    for (auto& elem : vals1) elem = dist(rd);
    for (auto& elem : vals2) elem = dist(rd);

    auto freq_vals1 = dft::dft(vals1);
    auto freq_vals2 = dft::dft(vals2);

    auto freq_mult = decltype(freq_vals1)(vals1.size(), 0);
    for (auto i = 0u; i < freq_vals1.size(); ++i)
        freq_mult[i] = freq_vals1[i] * freq_vals2[i];

    auto inv = dft::real(dft::inv_dft(freq_mult));

    std::reverse(vals2.begin(), vals2.end());
    auto expected = decltype(vals1)(vals1.size(), 0);
    for (auto i = 0u; i < vals1.size(); ++i)
        for (auto j = 0u; j < vals2.size(); ++j)
            expected[j] += vals1[j] * vals2[i];

    //for (auto i = 0u; i < vals1.size(); ++i)
        //std::cerr << inv[i] << " " << expected[i] << std::endl;
        //ASSERT_FLOAT_EQ(expected[i], inv[i]) << i;
}

