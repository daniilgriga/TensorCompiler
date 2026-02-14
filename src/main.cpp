#include <fstream>
#include <iostream>

#include "onnx.proto3.pb.h"

int main (int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: tc_main <model.onnx>" << std::endl;
        return 1;
    }

    onnx::ModelProto model;
    std::ifstream input (argv[1], std::ios::binary);

    if (!model.ParseFromIstream (&input))
    {
        std::cerr << "Error: failed to parse .onnx model" << std::endl;
        return 1;
    }

    const auto& graph = model.graph();
    std::cout << "model: " << graph.name() << std::endl
              << "nodes: " << graph.node_size() << std::endl
              << "input: " << graph.input_size() << std::endl
              << "output: " << graph.output_size() << std::endl
              << "init: " << graph.initializer_size() << std::endl;

    return 0;
}
