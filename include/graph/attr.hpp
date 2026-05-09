#pragma once

#include <cstdint>
#include <variant>
#include <string>
#include <unordered_map>
#include <vector>

namespace tc
{

    using AttrValue = std::variant<
        int64_t,
        float,
        std::string,
        std::vector<int64_t>,
        std::vector<float>,
        std::vector<std::string>
    >;

    using Attributes = std::unordered_map<std::string, AttrValue>;

} // namespace tc
