#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "importer/importer.hpp"
#include "onnx.proto3.pb.h"

namespace tc
{

namespace
{

onnx::ModelProto deserialize (const std::string& path)
{
    onnx::ModelProto model;

    std::ifstream input (path, std::ios::binary);
    if (!input.is_open())
        throw std::runtime_error ("import_onnx: cannot open file: " + path);

    if (!model.ParseFromIstream (&input))
        throw std::runtime_error ("import_onnx: failed to parse: " + path);

    return model;
}

void import_initializers (GraphBuilder& builder,
                          const onnx::GraphProto& graph,
                          std::unordered_set<std::string>& init_names)
{
    for (int i = 0; i < graph.initializer_size(); ++i)
    {
        const auto& init = graph.initializer (i);
        Value* val = builder.get_or_create_value (init.name());
        builder.mark_as_initializer (val);

        std::vector<int64_t> shape (init.dims().begin(), init.dims().end());
        val->set_shape (std::move (shape));

        init_names.insert (init.name());
    }
}

void import_inputs (GraphBuilder& builder,
                    const onnx::GraphProto& graph,
                    const std::unordered_set<std::string>& init_names)
{
    for (int i = 0; i < graph.input_size(); ++i)
    {
        const auto& input = graph.input (i);
        Value* val = builder.get_or_create_value (input.name());

        if (init_names.count (input.name()))
            continue;

        builder.mark_as_input (val);
    }
}

void import_outputs (GraphBuilder& builder,
                     const onnx::GraphProto& graph)
{
    for (int i = 0; i < graph.output_size(); ++i)
    {
        const auto& output = graph.output (i);
        Value* val = builder.get_or_create_value (output.name());
        builder.mark_as_output (val);
    }
}

void import_nodes (GraphBuilder& /*builder*/,
                   const onnx::GraphProto& /*graph*/)
{
    // graph.node[] -> add_node
}

} // namespace

GraphBuilder import_onnx (const std::string& path)
{
    auto model = deserialize (path);
    const auto& graph = model.graph ();

    GraphBuilder builder;
    std::unordered_set<std::string> init_names;

    import_initializers (builder, graph, init_names);
    import_inputs (builder, graph, init_names);
    import_outputs (builder, graph);
    import_nodes (builder, graph);

    return builder;
}

} // namespace tc
