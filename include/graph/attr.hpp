#pragma once

#include <cstdint>
#include <variant>
#include <string>
#include <unordered_map>
#include <vector>

namespace tc
{
    using AttrValue = std::variant<
    int64_t,                 // int
    float,                   // float
    std::string,             // string
    std::vector<int64_t>,    // ints
    std::vector<float>,      // floats
    std::vector<std::string> // strings
    >;

    using Attributes = std::unordered_map<std::string, AttrValue>;
} // namespace tc