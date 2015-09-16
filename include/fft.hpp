#pragma once

#include <complex>
#include <vector>
#include <iostream>
#include "matrix.hpp"

namespace fft
{

template <typename T>
using ComplexVec=std::vector<std::complex<T>>;

namespace impl
{

template <typename T, bool is_inverse, typename It, typename DestIt>
void fft_impl(It begin, It end, size_t step, DestIt dest)
{
    static T pi = T(3.141592653589793238463);
    static T pi2 = pi * 2.0;
    static std::complex<T> ci(0, 1);
    static std::complex<T> mult = (is_inverse) ? ci : -ci;
    size_t size = (end - begin) / step;
    if (size == 0) return;
    if (size == 1) {
        *dest = *begin;
        return;
    }
    assert(size % 2 == 0);
    size_t half = size / 2;
    
    auto next_step = 2 * step;
    fft_impl<T, is_inverse>(begin, begin + half * next_step, next_step, dest);
    fft_impl<T, is_inverse>(begin + step, begin + step + half * next_step, next_step, dest + half); 

    auto dest_it = dest;
    for (auto i = 0; i < half; ++i)
    {
        auto& dst = *dest_it;
        auto& odd = *(dest_it + half);
        dst = odd * exp(mult * T(i) * pi2 / T(size)) + dst;
        dest_it++;
    }
    for (auto i = half; i < size; ++i)
    {
        auto& dst = *dest_it;
        auto& prev = *(dest_it - half);
        dst = prev - dst * exp(mult * T(i - half) * pi2 / T(size)) + dst * exp(mult * T(i) * pi2 / T(size));

        dest_it++;
    }
    if (is_inverse and (step == 1))
        for (auto it = dest; it != dest + size; ++it)
            *it /= size;
}


template <bool is_inverse, typename T>
auto fft_impl(ComplexVec<T> input) -> decltype(input)
{
    ComplexVec<T> dest(input.size());
    fft_impl<T, is_inverse>(input.begin(), input.end(), 1u, dest.begin());
    return dest;
}

template <bool is_inverse, typename T>
ComplexVec<T> fft_2d_impl(ComplexVec<T> input, size_t width) 
{
    if (input.size() == 1) return input;
    
    auto height = input.size() / width;

    auto perform_row_fft = [](ComplexVec<T> input, size_t width, size_t height){
            ComplexVec<T> result;
            for (auto row = 0u; row < height; ++row)
            {
                ComplexVec<T> row_data(input.begin() + row * width, input.begin() + (row + 1) * width);
                auto row_result = fft_impl<is_inverse>(row_data);
                result.insert(result.end(), row_result.begin(), row_result.end());
            }
            return result;
        };
    
    auto ret = matrix::transpose(perform_row_fft(
        matrix::transpose(
            perform_row_fft(input, width, height), width), height, width), height);
    assert(ret.size() == input.size());
    return ret;
}

} //namespace impl

template <typename T>
ComplexVec<T> fft(ComplexVec<T> input)
{
    return impl::fft_impl<false>(input);
}

template <typename T>
auto inv_fft(ComplexVec<T> input) -> decltype(input)
{
    return impl::fft_impl<true>(input);
}

template <typename T>
auto fft_2d(ComplexVec<T> input, size_t width) -> decltype(input)
{
    return impl::fft_2d_impl<false>(input, width);
}

template <typename T>
auto inv_fft_2d(ComplexVec<T> input, size_t width) -> decltype(input)
{
    return impl::fft_2d_impl<true>(input, width);
}

} //namespace fft

