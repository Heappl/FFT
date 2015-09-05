#pragma once

#include <random>
#include <vector>

inline std::vector<double> generate(size_t size = 100,
                                    double from = -100.0,
                                    double to = 100.0)
{
    static std::mt19937 rd;

    std::uniform_real_distribution<double> dist(from, to);
    std::vector<double> ret(size);
    for (auto& elem : ret) elem = dist(rd);
    return ret;
}

