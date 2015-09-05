#pragma once

namespace fft
{
namespace impl
{

template <bool is_inverse, typename T>
auto fft_impl(std::vector<std::complex<T>> input) -> decltype(input)
{
    return input;
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

