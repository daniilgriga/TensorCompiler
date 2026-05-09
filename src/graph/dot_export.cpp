#include "graph/dot_export.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace
{

    std::string format_shape (const std::vector<int64_t>& shape)
    {
        if (shape.empty()) return "";

        std::string result;
        // use angle brackets like <1x1x3x3>
        result += "&lt;";
        for (std::size_t i = 0; i < shape.size(); ++i)
        {
            if (i > 0) result += "x";
            result += std::to_string (shape[i]);
        }
        result += "&gt;";
        return result;
    }

    std::string dot_id (const std::string& name)
    {
        std::string id;
        id.reserve (name.size());
        for (char c : name)
        {
            if (c == '/' || c == ':' || c == '.' || c == '-')
                id += '_';
            else
                id += c;
        }

        return id;
    }

    // map op_type to a color (similar to Netron palette)
    const char* op_color (const std::string& op_type)
    {
        if (op_type == "Conv")
            return "#3b5998";

        if (op_type == "Relu")
            return "#795548";

        if (op_type == "MaxPool" || op_type == "AveragePool" || op_type == "GlobalAveragePool")
            return "#388e3c";

        if (op_type == "Gemm" || op_type == "MatMul")
            return "#3b5998";

        if (op_type == "Reshape" || op_type == "Flatten" || op_type == "Squeeze" || op_type == "Unsqueeze")
            return "#5c6bc0";

        if (op_type == "Concat")
            return "#6d4c41";

        if (op_type == "BatchNormalization" || op_type == "InstanceNormalization")
            return "#00796b";

        if (op_type == "Softmax")
            return "#e65100";

        if (op_type == "Add" || op_type == "Mul" || op_type == "Sub" || op_type == "Div")
            return "#455a64";

        if (op_type == "Dropout")
            return "#7b1fa2";

        return "#546e7a"; // default gray-blue
    }

    // build a map: node_name -> list of initializer inputs
    using InitInfo = std::vector<std::pair<std::string, std::string>>; // (name, shape)
    using NodeInitMap = std::unordered_map<std::string, InitInfo>;

    NodeInitMap build_init_map (const tc::GraphBuilder& graph)
    {
        NodeInitMap result;
        for (const auto& node : graph.nodes())
        {
            InitInfo inits;
            for (const auto* val : node->inputs())
            {
                if (!val->is_initializer())
                    continue;

                std::string label;
                const auto& name = val->name();
                if (name.size() >= 2 && name[1] == '_')
                    label += name[0]; // W, B, ...
                else
                    label += name;

                std::string shape = format_shape (val->shape());
                if (!shape.empty()) label += " " + shape;

                inits.emplace_back (name, label);
            }
            if (!inits.empty())
                result[node->name()] = std::move (inits);
        }

        return result;
    }

    // collect all initializer names (to skip them as separate edge sources)
    std::unordered_set<std::string> collect_init_names (const tc::GraphBuilder& graph)
    {
        std::unordered_set<std::string> names;
        for (const auto* val : graph.graph_initializers())
            names.insert (val->name());

        return names;
    }

    void emit_graph_inputs (const tc::GraphBuilder& graph, std::ostream& out)
    {
        for (const auto* val : graph.graph_inputs())
        {
            out << "    \"input_" << dot_id (val->name()) << "\" ["
                << "label=\"" << val->name() << "\" "
                << "shape=ellipse "
                << "style=filled "
                << "fillcolor=\"#4a4a4a\" "
                << "fontcolor=white "
                << "color=\"#666666\" "
                << "penwidth=1.5"
                << "];\n";
        }
    }

    void emit_nodes (const tc::GraphBuilder& graph, const NodeInitMap& init_map,
                     std::ostream& out)
    {
        for (const auto& node : graph.nodes())
        {
            const auto* color = op_color (node->op_type());
            auto it = init_map.find (node->name());
            bool has_inits = it != init_map.end();

            // HTML label for two-part node (header + body)
            out << "    \"node_" << dot_id (node->name()) << "\" ["
                << "shape=plain label=<\n";

            out << "    <TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"8\">\n";

            // header row: op_type (colored background)
            out << "      <TR><TD BGCOLOR=\"" << color
                << "\" ALIGN=\"CENTER\"><FONT COLOR=\"white\"><B> "
                << node->op_type()
                << " </B></FONT></TD></TR>\n";

            // initializer rows (darker body)
            if (has_inits)
            {
                for (const auto& [name, label] : it->second)
                {
                    out << "      <TR><TD BGCOLOR=\"#3a3a3a\" ALIGN=\"LEFT\">"
                        << "<FONT COLOR=\"#cccccc\" POINT-SIZE=\"10\"><B> "
                        << label
                        << " </B></FONT></TD></TR>\n";
                }
            }

            out << "    </TABLE>\n";
            out << "    >];\n";
        }
    }

    void emit_graph_outputs (const tc::GraphBuilder& graph, std::ostream& out)
    {
        for (const auto* val : graph.graph_outputs())
        {
            out << "    \"output_" << dot_id (val->name()) << "\" ["
                << "label=\"" << val->name() << "\" "
                << "shape=ellipse "
                << "style=filled "
                << "fillcolor=\"#4a4a4a\" "
                << "fontcolor=white "
                << "color=\"#666666\" "
                << "penwidth=1.5"
                << "];\n";
        }
    }

    void emit_edges (const tc::GraphBuilder& graph,
                     const std::unordered_set<std::string>& init_names,
                     std::ostream& out)
    {
        for (const auto& node : graph.nodes())
        {
            for (const auto* val : node->inputs())
            {
                // skip initializers - they are shown inside the node
                if (init_names.count (val->name())) continue;

                std::string src;

                if (val->is_graph_input())
                    src = "input_" + dot_id (val->name());
                else if (val->producer())
                    src = "node_" + dot_id (val->producer()->name());
                else
                    continue;

                std::string dst = "node_" + dot_id (node->name());

                out << "    \"" << src << "\" -> \"" << dst << "\" ["
                    << "label=\" " << val->name() << " \" "
                    << "fontsize=9 "
                    << "fontcolor=\"#999999\""
                    << "];\n";
            }

            for (const auto* val : node->outputs())
            {
                if (!val->is_graph_output()) continue;

                std::string src = "node_" + dot_id (node->name());
                std::string dst = "output_" + dot_id (val->name());

                out << "    \"" << src << "\" -> \"" << dst << "\" ["
                    << "label=\" " << val->name() << " \" "
                    << "fontsize=9 "
                    << "fontcolor=\"#999999\""
                    << "];\n";
            }
        }
    }

} // namespace

namespace tc
{

    void dump_dot (const GraphBuilder& graph, std::ostream& out)
    {
        auto init_map = build_init_map (graph);
        auto init_names = collect_init_names (graph);

        out << "digraph G {\n"
            << "    rankdir=TB;\n"
            << "    bgcolor=\"#2b2b2b\";\n"
            << "    fontname=\"Helvetica\";\n"
            << "    nodesep=1.0;\n"
            << "    ranksep=0.8;\n"
            << "    node [fontname=\"Helvetica\" fontsize=12 fontcolor=white color=\"#555555\"];\n"
            << "    edge [fontname=\"Helvetica\" color=\"#888888\" "
            << "arrowhead=normal arrowsize=0.7 labeldistance=2.0 labelangle=0];\n"
            << "\n";

        emit_graph_inputs (graph, out);
        out << "\n";

        emit_nodes (graph, init_map, out);
        out << "\n";

        emit_graph_outputs (graph, out);
        out << "\n";

        emit_edges (graph, init_names, out);

        out << "}\n";
    }

} // namespace tc
