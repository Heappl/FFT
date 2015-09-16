#pragma once

#include <complex>
#include <vector>
#include "matrix.hpp"

namespace fft
{

template <typename T>
using ComplexVec=std::vector<std::complex<T>>;

namespace impl
{

template <bool is_inverse, bool is_first, typename T>
auto fft_impl(ComplexVec<T> input) -> decltype(input)
{
    static T pi = T(3.141592653589793238463);
    static T pi2 = pi * 2.0;
    static std::complex<T> ci(0, 1);
    static std::complex<T> mult = (is_inverse) ? ci : -ci;
    if (input.size() == 1) return input;
    assert(input.size() % 2 == 0);

    T N = input.size();
    auto even = decltype(input)(input.size() / 2);
    auto odd = decltype(input)(input.size() / 2);

    for (auto i = 0; i < even.size(); ++i)
    {
        even[i] = input[2 * i];
        odd[i] = input[2 * i + 1];
    }

    auto even_result = fft_impl<is_inverse, false>(even);
    auto odd_result = fft_impl<is_inverse, false>(odd);

    ComplexVec<T> result(input.size());
    for (auto i = 0; i < result.size(); ++i)
    {
        result[i] = odd_result[i % odd.size()] * exp(mult * T(i) * pi2 / N)
            + even_result[i % even.size()];
        if (is_inverse and is_first) result[i] /= N;
    }
    
    return result;
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
                auto row_result = fft_impl<is_inverse, true>(row_data);
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
    return impl::fft_impl<false, true>(input);
}

template <typename T>
auto inv_fft(ComplexVec<T> input) -> decltype(input)
{
    return impl::fft_impl<true, true>(input);
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

