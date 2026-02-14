#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
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

        bool contains_node_ptr (const Node* node) const
        {
            for (const auto& n_uptr : nodes_)
            {
                if (n_uptr.get() == node)
                    return true;
            }
            return false;
        }

        bool contains_value_ptr (const Value* value) const
        {
            for (const auto& v_uptr : values_)
            {
                if (v_uptr.get() == value)
                    return true;
            }
            return false;
        }

        static bool value_in_outputs (const Node* node, const Value* value)
        {
            for (Value* out : node->outputs())
            {
                if (out == value)
                    return true;
            }
            return false;
        }

        static bool node_in_consumers (const Value* value, const Node* node)
        {
            for (Node* consumer : value->consumers())
            {
                if (consumer == node)
                    return true;
            }
            return false;
        }

        void verify_values_consistency () const
        {
            for (const auto& v_uptr : values_)
            {
                const Value* value = v_uptr.get();
                const Node* producer = value->producer();
                if (!producer)
                    continue;

                if (!contains_node_ptr(producer))
                    throw std::runtime_error("[verify]: value has producer not in nodes_");

                if (!value_in_outputs(producer, value))
                    throw std::runtime_error("[verify]: producer doesn't have value in outputs");
            }
        }

        void verify_nodes_consistency () const
        {
            for (const auto& n_uptr : nodes_)
            {
                const Node* node = n_uptr.get();

                for (Value* in : node->inputs())
                {
                    if (!in)
                        throw std::runtime_error("[verify]: node input is nullptr");

                    if (!contains_value_ptr(in))
                        throw std::runtime_error("[verify]: node input value is not in values_");

                    if (!node_in_consumers(in, node))
                        throw std::runtime_error("[verify]: node missing in input consumers");
                }

                for (Value* out : node->outputs())
                {
                    if (!out)
                        throw std::runtime_error("[verify]: node output is nullptr");

                    if (!contains_value_ptr(out))
                        throw std::runtime_error("[verify]: node output value is not in values_");

                    if (out->producer() != node)
                        throw std::runtime_error("[verify]: node output has wrong producer");
                }
            }
        }

        void verify_graph_inputs () const
        {
            std::unordered_set<const Value*> seen;

            for (const Value* value : graph_inputs_)
            {
                if (!value)
                    throw std::runtime_error("[verify]: graph_inputs has nullptr value");

                if (!contains_value_ptr(value))
                    throw std::runtime_error("[verify]: graph_inputs value is not in values_");
                
                if (!value->is_graph_input())
                    throw std::runtime_error("[verify]: graph_inputs value is not marked as input");

                if (!seen.insert(value).second)
                    throw std::runtime_error("[verify]: duplicate value in graph_inputs");
            }
        }

        void verify_graph_outputs () const
        {
            std::unordered_set<const Value*> seen;

            for (const Value* value : graph_outputs_)
            {
                if (!value)
                    throw std::runtime_error("[verify]: graph_outputs has nullptr value");

                if (!contains_value_ptr(value))
                    throw std::runtime_error("[verify]: graph_outputs value is not in values_");

                if (!value->is_graph_output())
                    throw std::runtime_error("[verify]: graph_outputs value is not marked as output");

                if (!seen.insert(value).second)
                    throw std::runtime_error("[verify]: duplicate value in graph_outputs");
            }
        }

        void verify_graph_initializers () const
        {
            std::unordered_set<const Value*> seen;

            for (const Value* value : graph_initializers_)
            {
                if (!value)
                    throw std::runtime_error("[verify]: graph_initializers has nullptr value");

                if (!contains_value_ptr(value))
                    throw std::runtime_error("[verify]: graph_initializers value is not in values_");

                if (!value->is_initializer())
                    throw std::runtime_error("[verify]: graph_initializers value is not marked as initializer");

                if (!seen.insert(value).second)
                    throw std::runtime_error("[verify]: duplicate value in graph_initializers");
            }
        }

        void verify_unique_value_names () const
        {
            std::unordered_set<std::string> names;

            for (const auto& v_uptr : values_)
            {
                const Value* value = v_uptr.get();
                if (!value)
                    throw std::runtime_error("[verify]: values_ has nullptr");

                if (!names.insert(value->name()).second)
                    throw std::runtime_error("[verify]: duplicate value name: " + value->name());
            }
        }

        void verify_value_map_consistency () const
        {
            // map -> values_
            for (const auto& kv : value_map_)
            {
                const std::string& name = kv.first;
                const Value* value = kv.second;

                if (!value)
                    throw std::runtime_error("[verify]: value_map has nullptr value");

                if (!contains_value_ptr(value))
                    throw std::runtime_error("[verify]: value_map points to value not in values_");
                
                if (value->name() != name)
                    throw std::runtime_error("[verify]: value_map key/name mismatch for value");
            }

            //values_ -> map
            for (const auto& v_uptr : values_)
            {
                const Value* value = v_uptr.get();
                if (!value)
                    throw std::runtime_error("[verify]: values_ has nullptr");
                
                auto it = value_map_.find(value->name());
                if (it == value_map_.end())
                    throw std::runtime_error("[verify]: value from values_ is missing in value_map_");
                
                if (it->second != value)
                    throw std::runtime_error("[verify]: value_map points to different Value* for name");
            }
        }

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

        void verify() const
        {
            verify_values_consistency();
            verify_nodes_consistency();
            verify_graph_inputs();
            verify_graph_outputs();
            verify_graph_initializers();
            verify_unique_value_names();
            verify_value_map_consistency();
        }
    };
} // namespace tc
