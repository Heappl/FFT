#pragma once

#include <vector>
#include <stdexcept>

namespace matrix
{

template <typename T>
std::vector<T> transpose(std::vector<T> arg, size_t width)
{
    auto height = arg.size() / width;
    std::vector<T> ret(height * width);
    for (auto row = 0u; row < height; ++row)
        for (auto col = 0u; col < width; ++col)
            ret[col * height + row] = arg[row * width + col];
    return ret;
}

} //namespace matrix

