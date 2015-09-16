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
    auto aux_size = size;
    for (auto i = 0u; i < sizeof(aux_size) * 8 - 1; ++i)
    {
        if ((aux_size & (1 << i)) == 0) continue;
        aux_size -= 1 << i;
        if (aux_size == 0)
        {
            if (size == (1 << i)) return size;
            return (1 << (i + 1));
        }
    }
    throw std::runtime_error("the nearest power of two is too big");
}

template <typename T>
std::vector<T> resize_matrix(
    std::vector<T> arg,
    size_t original_width,
    size_t original_height,
    size_t height,
    size_t width,
    bool resize_to_nearest_power_of_2 = false)
{
    auto final_size = height * width;
    if (resize_to_nearest_power_of_2) final_size = nearest_power_of_2(final_size);
    std::vector<T> ret(final_size);
    auto copy_height = std::min(height, original_height);
    auto copy_width = std::min(width, original_width);
    for (auto row = 0u; row < copy_height; ++row)
        for (auto col = 0u; col < copy_width; ++col)
            ret[row * width + col] = arg[row * original_width + col];
    return ret;
}
} //namespace detail

template <typename T, bool T_resize = true>
std::vector<T> convolve(std::vector<T> first,
                        std::vector<T> second)
{
    auto original_size = first.size();
    if (T_resize)
    {
        auto calc_size = detail::nearest_power_of_2(first.size() + second.size());
        first.resize(calc_size, T());
        second.resize(first.size(), T());
    }
    std::reverse(second.begin(), second.end());
    std::rotate(second.begin(), second.end() - 1, second.end());

    auto freq_first = fft::fft(dft::real2complex(first));
    auto freq_second = fft::fft(dft::real2complex(second));

    for (auto i = 0u; i < first.size(); ++i)
        freq_first[i] *= freq_second[i];
    
    auto inverted = fft::inv_fft(freq_first);
    if (T_resize)
    {
        inverted.resize(original_size);
    }

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
    first = detail::resize_matrix(first, first_width, first_height, height, width, true);
    second = detail::resize_matrix(second, second_width, second_height, height, width, true);

    return detail::resize_matrix(
        convolve<T, false>(first, second),
        width, height, first_height, first_width);
}

} //namespace convolution

