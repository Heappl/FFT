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

