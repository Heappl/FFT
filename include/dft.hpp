#pragma once

#include <vector>
#include <complex>
#include <type_traits>
#include "matrix.hpp"

namespace dft
{

template <typename T>
using ComplexVec=std::vector<std::complex<T>>;

namespace impl
{

template <typename RandomAccessIt>
auto log_sum(RandomAccessIt begin, RandomAccessIt end)
    -> typename std::remove_reference<decltype(*begin)>::type
{
    if (end == begin) return decltype(log_sum(begin, end))();
    if (end - begin == 1) return *begin;

    auto middle = begin + (end - begin) / 2;
    auto first = log_sum(begin, middle);
    auto second = log_sum(middle, end);
    return first + second;
}

template <bool is_inverse, typename T>
auto dft_impl(ComplexVec<T> input) -> decltype(input)
{
    static T pi = T(3.141592653589793238463);
    static std::complex<T> i(0, 1);

    T N = input.size();
    ComplexVec<T> result(input.size(), 0);
    for (auto k = 0u; k < input.size(); ++k)
    {
        auto input_copy = input;

        for (auto n = 0u; n < input.size(); ++n)
        {
            auto power = i * T(2.0f) * pi * T(n * k) / N;
            if (not is_inverse) power = -power;
            input_copy[n] *= std::exp(power);
        }
        result[k] = log_sum(input_copy.begin(), input_copy.end());
        if (is_inverse) result[k] /= N;
    }
    return result;
}

template <bool is_inverse, typename T>
auto dft_2d_impl(ComplexVec<T> input, size_t width, size_t height) -> decltype(input)
{
    static T pi = T(3.141592653589793238463);
    static T pi2 = pi * 2.0;
    static std::complex<T> i(0, 1);

    T N1 = height;
    T N2 = width;
    ComplexVec<T> result(input.size());
    for (auto k1 = 0u; k1 < height; ++k1)
    {
        for (auto k2 = 0u; k2 < width; ++k2)
        {
            ComplexVec<T> input_copy = input;
            for (auto n1 = 0u; n1 < height; ++n1)
            {
                for (auto n2 = 0u; n2 < width; ++n2)
                {
                    auto power1 = i * pi2 * T(n1 * k1) / N1;
                    auto power2 = i * pi2 * T(n2 * k2) / N2;
                    if (not is_inverse)
                    {
                        power1 = -power1;
                        power2 = -power2;
                    }
                    input_copy[n1 * width + n2] *= std::exp(power1 + power2);
                }
            }
            result[k1 * width + k2] = log_sum(input_copy.begin(), input_copy.end());
            if (is_inverse) result[k1 * width + k2] /= N1 * N2;
        }
    }
    return result;
}

} //namespace impl

template <typename T>
auto dft(std::vector<std::complex<T>> input) -> decltype(input)
{
    return impl::dft_impl<false>(input);
}

template <typename T>
auto inv_dft(std::vector<std::complex<T>> input) -> decltype(input)
{
    return impl::dft_impl<true>(input);
}

template <typename T>
auto dft_2d(std::vector<std::complex<T>> input, size_t width) -> decltype(input)
{
    return impl::dft_2d_impl<false>(input, width, input.size() / width);
}

template <typename T>
auto inv_dft_2d(std::vector<std::complex<T>> input, size_t width) -> decltype(input)
{
    return impl::dft_2d_impl<true>(input, width, input.size() / width);
}

template <typename T>
ComplexVec<T> real2complex(const std::vector<T>& input)
{
    ComplexVec<T> ret(input.size());
    for (auto i = 0u; i < input.size(); ++i)
        ret[i] = std::complex<T>(input[i]);
    return ret;
}

template <typename T>
auto real(std::vector<std::complex<T>> input) -> std::vector<T>
{
    std::vector<T> ret(input.size());
    for (auto i = 0u; i < ret.size(); ++i)
        ret[i] = input[i].real();
    return ret;
}

} //namespace dft

