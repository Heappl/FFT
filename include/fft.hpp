#pragma once

namespace fft
{
namespace impl
{

template <bool is_inverse, typename T>
auto fft_impl(std::vector<std::complex<T>> input) -> decltype(input)
{
    static T pi = T(3.141592653589793238463);
    static T pi2 = pi * T(2);
    static std::complex<T> ci(0, 1);
    if (input.size() == 1) return input;

    T N = input.size();
    auto first = decltype(input)(input.size() / 2);
    auto second = decltype(input)(input.size() / 2);

    for (auto i = 0; i < first.size(); ++i)
    {
        first[i] = input[2 * i];
        second[i] = input[2 * i + 1];
    }

    auto first_result = fft_impl<is_inverse>(first);
    auto second_result = fft_impl<is_inverse>(second);

    auto result = decltype(input)(input.size());
    for (auto i = 0; i < result.size(); ++i)
        result[i] = first_result[i % first.size()] * exp(-ci * T(i) * pi2 / N)
            + second_result[i % second.size()];
    
    return result;
}
} //namespace impl

template <typename T>
auto fft(std::vector<std::complex<T>> input) -> decltype(input)
{
    return impl::fft_impl<false>(input);
}

template <typename T>
auto fft(std::vector<T> input) -> std::vector<std::complex<T>>
{
    std::vector<std::complex<T>> complex_input(input.size());
    for (auto i = 0u; i < input.size(); ++i)
        complex_input[i] = std::complex<T>(input[i]);
    return impl::fft_impl<false>(complex_input);
}

template <typename T>
auto inv_fft(std::vector<std::complex<T>> input) -> decltype(input)
{
    return impl::fft_impl<true>(input);
}

} //namespace fft

