#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <utility>

namespace tc
{
    class Node; // forward declaration

    class Value
    {
    private:
        std::string name_;
        std::vector<int64_t> shape_;

        bool is_graph_input_ = false;
        bool is_graph_output_ = false;
        bool is_initializer_ = false;

        Node* producer_ = nullptr;
        std::vector<Node*> consumers_;

        explicit Value(std::string name)
        : name_(std::move(name)) {}

    public:

        const std::string& name () const { return name_; }
        const std::vector<int64_t>& shape () const { return shape_; }

        bool is_graph_input () const { return is_graph_input_; }
        bool is_graph_output () const { return is_graph_output_; }
        bool is_initializer () const { return is_initializer_; }

        Node* producer () const { return producer_; }
        const std::vector<Node*>& consumers () const  { return consumers_; }

        void set_shape (std::vector<int64_t> shape) { shape_ = std::move(shape); }

    private:
        friend class GraphBuilder;

        void mark_input () { is_graph_input_ = true; }
        void mark_output () { is_graph_output_ = true; }
        void mark_initializer () { is_initializer_ = true; }

        void set_producer (Node* n) { producer_ = n; }
        void add_consumer (Node* n) {consumers_.push_back(n); }
    };
} // namespace tc
