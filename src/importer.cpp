#include <cstring>
#include <iostream>
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

        // pack typed repeated fields into raw bytes for uniform downstream use
        template <typename T>
        std::vector<uint8_t> pack_as_raw (const google::protobuf::RepeatedField<T>& field)
        {
            std::vector<uint8_t> raw (field.size() * sizeof(T));
            for (int i = 0; i < field.size(); ++i)
            {
                T v = field[i];
                std::memcpy (raw.data() + i * sizeof(T), &v, sizeof(T));
            }
            return raw;
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

                builder.set_value_dtype (val, static_cast<DType> (init.data_type()));

                // normalize to raw bytes regardless of which storage field ONNX uses
                if (!init.raw_data().empty())
                {
                    const auto& rd = init.raw_data();
                    builder.set_value_data (val, std::vector<uint8_t> (rd.begin(), rd.end()));
                }
                else
                {
                    switch (init.data_type())
                    {
                        case onnx::TensorProto::FLOAT:
                            builder.set_value_data (val, pack_as_raw (init.float_data()));
                            break;
                        case onnx::TensorProto::INT32:
                            builder.set_value_data (val, pack_as_raw (init.int32_data()));
                            break;
                        case onnx::TensorProto::INT64:
                            builder.set_value_data (val, pack_as_raw (init.int64_data()));
                            break;
                        case onnx::TensorProto::DOUBLE:
                            builder.set_value_data (val, pack_as_raw (init.double_data()));
                            break;
                        default:
                            std::cerr << "warning: no raw_data and unsupported typed field "
                                      << "for initializer '" << init.name() << "'\n";
                            break;
                    }
                }

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

                if (input.type().has_tensor_type())
                {
                    const auto& tt = input.type().tensor_type();

                    if (tt.elem_type() != 0)
                        builder.set_value_dtype (val, static_cast<DType> (tt.elem_type()));

                    if (tt.has_shape())
                    {
                        std::vector<int64_t> shape;
                        shape.reserve (tt.shape().dim_size());
                        for (int d = 0; d < tt.shape().dim_size(); ++d)
                            shape.push_back (tt.shape().dim(d).dim_value());
                        val->set_shape (std::move (shape));
                    }
                }
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

        Attributes parse_attributes (const onnx::NodeProto& onnx_node)
        {
            Attributes attrs;

            for (int i = 0; i < onnx_node.attribute_size(); ++i)
            {
                const auto& attr = onnx_node.attribute (i);

                switch (attr.type())
                {
                    case onnx::AttributeProto::FLOAT:
                        attrs[attr.name()] = attr.f();
                        break;

                    case onnx::AttributeProto::INT:
                        attrs[attr.name()] = attr.i();
                        break;

                    case onnx::AttributeProto::STRING:
                        attrs[attr.name()] = attr.s();
                        break;

                    case onnx::AttributeProto::FLOATS:
                        attrs[attr.name()] = std::vector<float> (attr.floats().begin(), attr.floats().end());
                        break;

                    case onnx::AttributeProto::INTS:
                        attrs[attr.name()] = std::vector<int64_t> (attr.ints().begin(), attr.ints().end());
                        break;

                    case onnx::AttributeProto::STRINGS:
                    {
                        std::vector<std::string> strs (attr.strings().begin(), attr.strings().end());
                        attrs[attr.name()] = std::move (strs);
                        break;
                    }

                    default:
                        std::cerr << "warning: unsupported attribute type for '"
                                  << attr.name () << "' in node '"
                                  << onnx_node.name () << "'\n";
                        break;
                }
            }

            return attrs;
        }

        void import_nodes (GraphBuilder& builder,
                           const onnx::GraphProto& graph)
        {
            for (int node_idx = 0; node_idx < graph.node_size(); ++node_idx)
            {
                const auto& onnx_node = graph.node (node_idx);

                std::vector<Value*> inputs;
                inputs.reserve (static_cast<std::size_t> (onnx_node.input_size()));

                bool seen_empty = false;
                for (int i = 0; i < onnx_node.input_size(); ++i)
                {
                    const auto& name = onnx_node.input (i);
                    if (name.empty())
                    {
                        seen_empty = true;
                        continue;
                    }
                    if (seen_empty)
                        throw std::runtime_error (
                            "import_nodes: non-trailing empty input in node '" +
                            onnx_node.name() + "' (op '" + onnx_node.op_type() +
                            "'): input " + std::to_string (i) + " ('" + name +
                            "') comes after an empty input slot");
                    inputs.push_back (builder.get_or_create_value (name));
                }

                std::vector<Value*> outputs;
                for (int i = 0; i < onnx_node.output_size(); ++i)
                    outputs.push_back (builder.get_or_create_value (onnx_node.output (i)));

                Attributes attrs = parse_attributes (onnx_node);

                builder.add_node (onnx_node.op_type(),
                                  std::move (inputs),
                                  std::move (outputs),
                                  std::move (attrs),
                                  onnx_node.name());
            }
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

        builder.verify();

        return builder;
    }

} // namespace tc
