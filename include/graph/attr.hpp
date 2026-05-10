#pragma once

#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
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

    namespace detail
    {
        template <typename T, typename Variant>
        struct is_variant_alternative : std::false_type {};

        template <typename T, typename... Ts>
        struct is_variant_alternative<T, std::variant<Ts...>>
            : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};
    } // namespace detail

    // T must be one of the alternatives stored in AttrValue
    template <typename T>
    concept AttrAlternative = detail::is_variant_alternative<T, AttrValue>::value;

} // namespace tc
