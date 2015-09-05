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

} //namespace convolution

