#include <gtest/gtest.h>
#include "convolution.hpp"
#include "generator.hpp"
#include "equality_checks.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <chrono>

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

template <typename T>
std::vector<T> naive_convolve_2d(std::vector<T> first, size_t first_width,
                                 std::vector<T> second, size_t second_width)
{
    size_t first_height = first.size() / first_width;
    size_t second_height = second.size() / second_width;
    std::vector<T> ret(first.size(), 0);

    for (auto y = 0u; y < first_height; ++y)
    {
        for (auto kern_y = 0u; kern_y < second_height; ++kern_y)
        {
            if (y + kern_y >= first_height) break;
            for (auto x = 0u; x < first_width; ++x)
            {
                for (auto kern_x = 0u; kern_x < second_width; ++kern_x)
                {
                    if (x + kern_x >= first_width) break;
                    ret[y * first_width + x] +=
                        first[(y + kern_y) * first_width + x + kern_x]
                            * second[kern_y * second_width + kern_x];
                }
            }
        }
    }
    return ret;
}

TEST(ConvolutionTest, simple_2d_convolution)
{
    for (auto arg_width : {11u, 16u})
    {
        for (auto kern_width : {2u, 3u, 5u, 8u})
        {
            if (kern_width >= arg_width) continue;
            for (auto arg_height : {11u, 16u})
            {
                for (auto kern_height : {2u, 3u, 5u, 8u})
                {
                    if (kern_height >= arg_height) continue;
                    auto vals = generate(arg_height * arg_width);
                    auto filter = generate(kern_width * kern_height);

                    auto expected = naive_convolve_2d(vals, arg_width, filter, kern_width);
                    auto result = convolution::convolve_2d(vals, arg_width, filter, kern_width);

                    ASSERT_NO_FATAL_FAILURE(equal(expected, result))
                        << "sizes: " << arg_height << "x" << arg_width << " and "
                                     << kern_height << "x" << kern_width;
                }
            }
        }
    }
}

TEST(ConvolutionTest, DISABLED_big_2d_convolution)
{
    auto height = 800u;
    auto width = 800u;
    auto kern_width = 199u;
    auto kern_height = 199u;
    auto vals = generate(height * width);
    auto filter = generate(kern_width * kern_height);

    auto t1 = std::chrono::system_clock::now();
    auto expected = naive_convolve_2d(vals, width, filter, kern_width);
    auto t2 = std::chrono::system_clock::now();
    auto result = convolution::convolve_2d(vals, width, filter, kern_width);
    auto t3 = std::chrono::system_clock::now();

    auto elapsed_naive = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    auto elapsed_fft = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2);
    std::cerr << "elapsed naive: " << elapsed_naive.count() << std::endl;
    std::cerr << "elapsed fft: " << elapsed_fft.count() << std::endl;

    ASSERT_NO_FATAL_FAILURE(equal(expected, result));
}

