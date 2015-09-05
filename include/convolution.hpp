#pragma once

#include <vector>
#include <algorithm>
#include "dft.hpp"

namespace convolution
{

template <typename T>
std::vector<T> convolve(std::vector<T> first,
                        std::vector<T> second)
{
    auto original_size = first.size();
    first.resize(first.size() + second.size(), T());
    second.resize(first.size(), T());
    std::reverse(second.begin(), second.end());
    std::rotate(second.begin(), second.begin() + second.size() - 1, second.end());

    auto freq_first = dft::dft(first);
    auto freq_second = dft::dft(second);

    for (auto i = 0u; i < first.size(); ++i)
        freq_first[i] *= freq_second[i];
    
    auto inverted = dft::inv_dft(freq_first);
    inverted.resize(original_size);

    return dft::real(inverted);
}

} //namespace convolution

