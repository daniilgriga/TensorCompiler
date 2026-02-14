#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cstddef>
#include <utility>

#include "node.hpp"
#include "value.hpp"

namespace tc
{
    class GraphBuilder 
    {
    private:
        std::vector<std::unique_ptr<Node>> nodes_;
        std::vector<std::unique_ptr<Value>> values_;

        std::unordered_map<std::string, Value*> value_map_;

        std::vector<Value*> graph_inputs_;
        std::vector<Value*> graph_outputs_;
        std::vector<Value*> graph_initializers_;

        std::size_t auto_node_counter_ = 0;

        bool contains_node_ptr (const Node* node) const;
        bool contains_value_ptr (const Value* value) const;

        static bool value_in_outputs (const Node* node, const Value* value);
        static bool node_in_consumers (const Value* value, const Node* node);

        void verify_values_consistency () const;
        void verify_nodes_consistency () const;
        void verify_graph_inputs () const;
        void verify_graph_outputs () const;
        void verify_graph_initializers () const;
        void verify_unique_value_names () const;
        void verify_value_map_consistency () const;

    public:
        GraphBuilder() = default;
        ~GraphBuilder() = default;

        GraphBuilder(const GraphBuilder&) = delete;
        GraphBuilder& operator=(const GraphBuilder&) = delete;

        GraphBuilder(GraphBuilder&&) noexcept = default;
        GraphBuilder& operator=(GraphBuilder&&) noexcept = default;

        Value* get_or_create_value (const std::string& name)
        {
            auto it = value_map_.find(name);
            if (it != value_map_.end()) return it->second;

            values_.push_back(std::unique_ptr<Value>(new Value(name)));
            Value* v = values_.back().get();
            value_map_.emplace(name, v);
            return v;
        }

        Value* find_value (const std::string& name)
        {
            auto it = value_map_.find(name);
            return (it == value_map_.end()) ? nullptr : it->second;
        }

        const Value* find_value (const std::string& name) const
        {
            auto it = value_map_.find(name);
            return (it == value_map_.end()) ? nullptr : it->second;
        }

        void mark_as_input (Value* value)
        {
            if (!value) return;

            value->mark_input();
            if (std::find(graph_inputs_.begin(), graph_inputs_.end(), value) == graph_inputs_.end())
                graph_inputs_.push_back(value);
        }

        void mark_as_output (Value* value)
        {
            if (!value) return;

            value->mark_output();
            if (std::find(graph_outputs_.begin(), graph_outputs_.end(), value) == graph_outputs_.end())
                graph_outputs_.push_back(value);
        }

        void mark_as_initializer (Value* value)
        {
            if (!value) return;

            value->mark_initializer();
            if (std::find(graph_initializers_.begin(), graph_initializers_.end(), value) == graph_initializers_.end())
                graph_initializers_.push_back(value);
        }

        Node* add_node (std::string op_type,
               std::vector<Value*> inputs,
               std::vector<Value*> outputs,
               Attributes attrs = {},
               std::string name = "")
        {
            for (auto* v : inputs)
                if (!v) throw std::runtime_error("[add_node]: nullptr in inputs");

            for (auto* v : outputs)
                if (!v) throw std::runtime_error("[add_node]: nullptr in outputs");

            if (name.empty())
                name = "n" + std::to_string(auto_node_counter_++);

            nodes_.push_back(std::unique_ptr<Node>(new Node(
                std::move(op_type),
                std::move(name),
                std::move(inputs),
                std::move(outputs),
                std::move(attrs)
                )));
            Node* node = nodes_.back().get();

            for (Value* in : node->inputs())
                in->add_consumer(node);

            for (Value* out : node->outputs())
            {
                if (out->producer() != nullptr)
                    throw std::runtime_error("[add_node]: output already has producer");
                out->set_producer(node);
            }

            return node;
        }

        const std::vector<std::unique_ptr<Node>>& nodes () const { return nodes_; }
        const std::vector<std::unique_ptr<Value>>& values () const { return values_; }

        const std::vector<Value*>& graph_inputs () const { return graph_inputs_; }
        const std::vector<Value*>& graph_outputs () const { return graph_outputs_; }
        const std::vector<Value*>& graph_initializers () const { return graph_initializers_; }

        void verify() const;
    };
} // namespace tc
