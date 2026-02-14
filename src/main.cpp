#include <iostream>

#include "importer/importer.hpp"

int main (int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: tc_main <model.onnx>" << std::endl;
        return 1;
    }

    try
    {
        tc::GraphBuilder builder = tc::import_onnx (argv[1]);

        std::cout << "nodes: "        << builder.nodes().size() << std::endl
                  << "values: "       << builder.values().size() << std::endl
                  << "inputs: "       << builder.graph_inputs().size() << std::endl
                  << "outputs: "      << builder.graph_outputs().size() << std::endl
                  << "initializers: " << builder.graph_initializers().size() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
