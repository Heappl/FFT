#pragma once

#include <vector>
#include <stdexcept>

namespace matrix
{
template <typename T>
class matrix
{
    std::vector<T> data;
    size_t width_;
    size_t height_;
public:
    matrix(size_t width = 0, size_t height = 0, T elem = T())
        : data(width * height, elem)
        , width_(width)
        , height_(height)
    {}

    matrix(std::vector<T> data, size_t width)
        : data(data)
        , width_(width)
        , height_(data.size() / width)
    {
        if (data.size() % width != 0)
            throw std::runtime_error("matrix construction failed - data is not a rectangle");
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }
    size_t size() const { return height_ * width_; }

    decltype(data)& raw() { return data; }
    const decltype(data)& raw() const { return data; }

    T& at(size_t row, size_t col) { return data[row * width_ + col]; }
    const T& at(size_t row, size_t col) const { return data[row * width_ + col]; }

    typename decltype(data)::iterator begin() { return data.begin(); }
    typename decltype(data)::const_iterator begin() const { return data.begin(); }
    typename decltype(data)::iterator end() { return data.end(); }
    typename decltype(data)::const_iterator end() const { return data.end(); }
};

template <typename T>
matrix<T> make_matrix(const std::vector<T>& data, size_t width)
{
    return matrix<T>(data, width);
}

} //namespace matrix

