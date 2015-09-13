#pragma once

#include <vector>
#include <algorithm>
#include "fft.hpp"
#include "dft.hpp"

namespace convolution
{

namespace detail
{

size_t nearest_power_of_2(size_t size)
{
    for (auto i = 0u; i < sizeof(size) * 8 - 1; ++i)
    {
        if ((size & (1 << i)) == 0) continue;
        size -= 1 << i;
        if (size == 0) return (1 << (i + 1));
    }
    throw std::runtime_error("the nearest power of two is too big");
}

template <typename T>
std::vector<T> resize_matrix(
    std::vector<T> arg,
    size_t original_width,
    size_t height,
    size_t width,
    T fill_with = T())
{
    std::vector<T> ret(height * width, fill_with);
    auto original_height = arg.size() / original_width;
    auto copy_height = std::min(height, original_height);
    auto copy_width = std::min(width, original_width);
    for (auto row = 0u; row < copy_height; ++row)
        for (auto col = 0u; col < copy_width; ++col)
            ret[row * width + col] = arg[row * original_width + col];
    return ret;
}

template <typename T>
std::vector<T> revert_matrix(std::vector<T> arg, size_t width)
{
    auto height = arg.size() / width;
    std::vector<T> ret(height * width);
    for (auto row = 0u; row < height; ++row)
        for (auto col = 0u; col < width; ++col)
            ret[(height - row - 1) * width + (width - col - 1)] = arg[row * width + col];
    return ret;
}

template <typename T>
std::vector<T> rotate_matrix_by_one(std::vector<T> arg, size_t width)
{
    auto height = arg.size() / width;
    std::vector<T> ret(height * width);
    for (auto row = 0u; row < height; ++row)
        for (auto col = 0u; col < width; ++col)
            ret[(row + height + 1) % height * width + (col + width + 1) % width] =
                arg[row * width + col];
    return ret;
}

} //namespace detail

template <typename T>
std::vector<T> convolve(std::vector<T> first,
                        std::vector<T> second)
{
    auto original_size = first.size();
    auto calc_size = detail::nearest_power_of_2(first.size() + second.size());
    first.resize(calc_size, T());
    second.resize(first.size(), T());
    std::reverse(second.begin(), second.end());
    std::rotate(second.begin(), second.end() - 1, second.end());

    auto freq_first = fft::fft(first);
    auto freq_second = fft::fft(second);

    for (auto i = 0u; i < first.size(); ++i)
        freq_first[i] *= freq_second[i];
    
    auto inverted = fft::inv_fft(freq_first);
    inverted.resize(original_size);

    return dft::real(inverted);
}

template <typename T>
std::vector<T> convolve_2d(
    std::vector<T> first, size_t first_width,
    std::vector<T> second, size_t second_width)
{
    auto first_height = first.size() / first_width;
    auto second_height = second.size() / second_width;

    if ((first_width < second_width) or (first_height < second_height))
        throw std::runtime_error("kernel dimension is bigger than input's");

    auto height = first_height + second_height;
    auto width = first_width + second_width;

    first = detail::resize_matrix(first, first_width, height, width);
    second = detail::resize_matrix(second, second_width, height, width);
    second = detail::revert_matrix(second, width);
    second = detail::rotate_matrix_by_one(second, width);

    auto freq_first = dft::dft_2d(dft::real2complex(first), width);
    auto freq_second = dft::dft_2d(dft::real2complex(second), width);

    for (auto i = 0u; i < first.size(); ++i)
        freq_first[i] *= freq_second[i];
    
    return detail::resize_matrix(dft::real(dft::inv_dft_2d(freq_first, width)),
                                 width, first_height, first_width);
}

} //namespace convolution

