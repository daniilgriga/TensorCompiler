#pragma once

#include <string>
#include "graph/graph_builder.hpp"

namespace tc
{
    // reads an .onnx file, parses it via protobuf,
    // and builds a GraphBuilder with all nodes, values, and connections
    GraphBuilder import_onnx (const std::string& path);

} // namespace tc
