#include "graph/graph_builder.hpp"

#include <stdexcept>
#include <unordered_set>

namespace tc
{
bool GraphBuilder::contains_node_ptr (const Node* node) const
{
    for (const auto& n_uptr : nodes_)
    {
        if (n_uptr.get() == node)
        {
            return true;
        }
    }
    return false;
}

bool GraphBuilder::contains_value_ptr (const Value* value) const
{
    for (const auto& v_uptr : values_)
    {
        if (v_uptr.get() == value)
        {
            return true;
        }
    }
    return false;
}

bool GraphBuilder::value_in_outputs (const Node* node, const Value* value)
{
    for (Value* out : node->outputs())
    {
        if (out == value)
        {
            return true;
        }
    }
    return false;
}

bool GraphBuilder::node_in_consumers (const Value* value, const Node* node)
{
    for (Node* consumer : value->consumers())
    {
        if (consumer == node)
        {
            return true;
        }
    }
    return false;
}

void GraphBuilder::verify_values_consistency () const
{
    for (const auto& v_uptr : values_)
    {
        const Value* value = v_uptr.get();
        const Node* producer = value->producer();
        if (!producer)
        {
            continue;
        }

        if (!contains_node_ptr(producer))
        {
            throw std::runtime_error(
                "[verify]: value '" + value->name() + "' has producer '"
                + producer->name() + "' not present in nodes_");
        }

        if (!value_in_outputs(producer, value))
        {
            throw std::runtime_error(
                "[verify]: producer '" + producer->name() + "' (op=" + producer->op_type()
                + ") does not list value '" + value->name() + "' in outputs");
        }
    }
}

void GraphBuilder::verify_nodes_consistency () const
{
    for (const auto& n_uptr : nodes_)
    {
        const Node* node = n_uptr.get();

        std::size_t input_idx = 0;
        for (Value* in : node->inputs())
        {
            if (!in)
            {
                throw std::runtime_error(
                    "[verify]: node '" + node->name() + "' (op=" + node->op_type()
                    + ") has nullptr in inputs at index " + std::to_string(input_idx));
            }

            if (!contains_value_ptr(in))
            {
                throw std::runtime_error(
                    "[verify]: node '" + node->name() + "' (op=" + node->op_type()
                    + ") input '" + in->name() + "' is not present in values_");
            }

            if (!node_in_consumers(in, node))
            {
                throw std::runtime_error(
                    "[verify]: input value '" + in->name()
                    + "' does not list node '" + node->name() + "' in consumers");
            }
            ++input_idx;
        }

        std::size_t output_idx = 0;
        for (Value* out : node->outputs())
        {
            if (!out)
            {
                throw std::runtime_error(
                    "[verify]: node '" + node->name() + "' (op=" + node->op_type()
                    + ") has nullptr in outputs at index " + std::to_string(output_idx));
            }

            if (!contains_value_ptr(out))
            {
                throw std::runtime_error(
                    "[verify]: node '" + node->name() + "' (op=" + node->op_type()
                    + ") output '" + out->name() + "' is not present in values_");
            }

            if (out->producer() != node)
            {
                throw std::runtime_error(
                    "[verify]: output value '" + out->name() + "' has wrong producer; expected '"
                    + node->name() + "'");
            }
            ++output_idx;
        }
    }
}

void GraphBuilder::verify_graph_inputs () const
{
    std::unordered_set<const Value*> seen;

    for (const Value* value : graph_inputs_)
    {
        if (!value)
        {
            throw std::runtime_error("[verify]: graph_inputs has nullptr value");
        }

        if (!contains_value_ptr(value))
        {
            throw std::runtime_error(
                "[verify]: graph_inputs contains value '" + value->name()
                + "' that is not present in values_");
        }

        if (!value->is_graph_input())
        {
            throw std::runtime_error("[verify]: graph_inputs value is not marked as input");
        }

        if (!seen.insert(value).second)
        {
            throw std::runtime_error("[verify]: duplicate value in graph_inputs");
        }
    }
}

void GraphBuilder::verify_graph_outputs () const
{
    std::unordered_set<const Value*> seen;

    for (const Value* value : graph_outputs_)
    {
        if (!value)
        {
            throw std::runtime_error("[verify]: graph_outputs has nullptr value");
        }

        if (!contains_value_ptr(value))
        {
            throw std::runtime_error(
                "[verify]: graph_outputs contains value '" + value->name()
                + "' that is not present in values_");
        }

        if (!value->is_graph_output())
        {
            throw std::runtime_error("[verify]: graph_outputs value is not marked as output");
        }

        if (!seen.insert(value).second)
        {
            throw std::runtime_error("[verify]: duplicate value in graph_outputs");
        }
    }
}

void GraphBuilder::verify_graph_initializers () const
{
    std::unordered_set<const Value*> seen;

    for (const Value* value : graph_initializers_)
    {
        if (!value)
        {
            throw std::runtime_error("[verify]: graph_initializers has nullptr value");
        }

        if (!contains_value_ptr(value))
        {
            throw std::runtime_error(
                "[verify]: graph_initializers contains value '" + value->name()
                + "' that is not present in values_");
        }

        if (!value->is_initializer())
        {
            throw std::runtime_error(
                "[verify]: graph_initializers value is not marked as initializer");
        }

        if (!seen.insert(value).second)
        {
            throw std::runtime_error("[verify]: duplicate value in graph_initializers");
        }
    }
}

void GraphBuilder::verify_unique_value_names () const
{
    std::unordered_set<std::string> names;

    for (const auto& v_uptr : values_)
    {
        const Value* value = v_uptr.get();
        if (!value)
        {
            throw std::runtime_error("[verify]: values_ has nullptr");
        }

        if (!names.insert(value->name()).second)
        {
            throw std::runtime_error("[verify]: duplicate value name: " + value->name());
        }
    }
}

void GraphBuilder::verify_value_map_consistency () const
{
    for (const auto& kv : value_map_)
    {
        const std::string& name = kv.first;
        const Value* value = kv.second;

        if (!value)
        {
            throw std::runtime_error("[verify]: value_map has nullptr value");
        }

        if (!contains_value_ptr(value))
        {
            throw std::runtime_error("[verify]: value_map points to value not in values_");
        }

        if (value->name() != name)
        {
            throw std::runtime_error(
                "[verify]: value_map key/name mismatch: key='" + name + "', value->name()='"
                + value->name() + "'");
        }
    }

    for (const auto& v_uptr : values_)
    {
        const Value* value = v_uptr.get();
        if (!value)
        {
            throw std::runtime_error("[verify]: values_ has nullptr");
        }

        auto it = value_map_.find(value->name());
        if (it == value_map_.end())
        {
            throw std::runtime_error(
                "[verify]: value '" + value->name() + "' from values_ is missing in value_map_");
        }

        if (it->second != value)
        {
            throw std::runtime_error(
                "[verify]: value_map points to different Value* for name");
        }
    }
}

void GraphBuilder::verify () const
{
    verify_values_consistency();
    verify_nodes_consistency();
    verify_graph_inputs();
    verify_graph_outputs();
    verify_graph_initializers();
    verify_unique_value_names();
    verify_value_map_consistency();
}

} // namespace tc
