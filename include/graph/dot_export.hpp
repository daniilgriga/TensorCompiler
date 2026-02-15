#pragma once

#include <ostream>

#include "graph_builder.hpp"

namespace tc
{
    void dump_dot (const GraphBuilder& graph, std::ostream& out);
} // namespace tc
