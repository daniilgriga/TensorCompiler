#pragma once

#include <string>
#include <vector>
#include <utility>

#include "value.hpp"
#include "attr.hpp"

namespace tc
{
    class Node
    {
    private:
        std::string op_type_;
        std::string name_;

        std::vector<Value*> inputs_;
        std::vector<Value*> outputs_;

        Attributes attributes_;

        Node(std::string op_type,
            std::string name,
            std::vector<Value*> inputs,
            std::vector<Value*> outputs,
            Attributes attributes)
            : op_type_(std::move(op_type)),
              name_(std::move(name)),
              inputs_(std::move(inputs)),
              outputs_(std::move(outputs)),
              attributes_(std::move(attributes))
        {}

        friend class GraphBuilder;

    public:
        const std::string& op_type () const { return op_type_; }
        const std::string& name () const { return name_; }

        const std::vector<Value*>& inputs () const { return inputs_; }
        const std::vector<Value*>& outputs () const { return outputs_; }

        const Attributes& attributes () const { return attributes_; }

        const AttrValue* attribute (const std::string& key) const
        {
            auto it = attributes_.find(key);
            return (it == attributes_.end()) ? nullptr : &it->second;
        }
    };
} // namespace tc
