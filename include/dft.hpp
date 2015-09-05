#pragma once

#include <vector>
#include <complex>
#include <type_traits>

namespace dft
{

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
auto dft_impl(std::vector<std::complex<T>> input) -> decltype(input)
{
    static T pi = T(3.141592653589793238463);
    static std::complex<T> i(0, 1);

    T N = input.size();
    decltype(input) result(input.size(), 0);
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
} //namespace impl

template <typename T>
auto dft(std::vector<std::complex<T>> input) -> decltype(input)
{
    return impl::dft_impl<false>(input);
}

template <typename T>
auto dft(std::vector<T> input) -> std::vector<std::complex<T>>
{
    std::vector<std::complex<T>> complex_input(input.size());
    for (auto i = 0u; i < input.size(); ++i)
        complex_input[i] = std::complex<T>(input[i]);
    return impl::dft_impl<false>(complex_input);
}

template <typename T>
auto inv_dft(std::vector<std::complex<T>> input) -> decltype(input)
{
    return impl::dft_impl<true>(input);
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

