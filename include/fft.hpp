#pragma once

namespace fft
{
namespace impl
{

template <bool is_inverse, bool is_first, typename T>
auto fft_impl(std::vector<std::complex<T>> input) -> decltype(input)
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

    auto result = decltype(input)(input.size());
    for (auto i = 0; i < result.size(); ++i)
    {
        result[i] = odd_result[i % odd.size()] * exp(mult * T(i) * pi2 / N)
            + even_result[i % even.size()];
        if (is_inverse and is_first) result[i] /= N;
    }
    
    return result;
}
} //namespace impl

template <typename T>
auto fft(std::vector<std::complex<T>> input) -> decltype(input)
{
    return impl::fft_impl<false, true>(input);
}

template <typename T>
auto fft(std::vector<T> input) -> std::vector<std::complex<T>>
{
    std::vector<std::complex<T>> complex_input(input.size());
    for (auto i = 0u; i < input.size(); ++i)
        complex_input[i] = std::complex<T>(input[i]);
    return impl::fft_impl<false, true>(complex_input);
}

template <typename T>
auto inv_fft(std::vector<std::complex<T>> input) -> decltype(input)
{
    return impl::fft_impl<true, true>(input);
}

} //namespace fft

