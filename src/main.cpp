#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "importer/importer.hpp"
#include "graph/dot_export.hpp"

namespace
{

    void print_usage ()
    {
        std::cerr << "Usage:\n"
                  << "  tc_main <model.onnx>               print graph stats\n"
                  << "  tc_main <model.onnx> --dot <file>  export graph to Graphviz dot\n";
    }

} // namespace

int main (int argc, char* argv[])
{
    if (argc < 2)
    {
        print_usage();
        return 1;
    }

    try
    {
        tc::GraphBuilder builder = tc::import_onnx (argv[1]);

        // --dot <file>
        if (argc >= 4 && std::strcmp (argv[2], "--dot") == 0)
        {
            std::ofstream ofs (argv[3]);
            if (!ofs)
            {
                std::cerr << "Error: cannot open " << argv[3] << std::endl;
                return 1;
            }

            tc::dump_dot (builder, ofs);
            std::cout << "dot exported to " << argv[3] << std::endl;

            return 0;
        }

        // default: print stats
        std::cout << "nodes:        " << builder.nodes().size() << "\n"
                  << "values:       " << builder.values().size() << "\n"
                  << "inputs:       " << builder.graph_inputs().size() << "\n"
                  << "outputs:      " << builder.graph_outputs().size() << "\n"
                  << "initializers: " << builder.graph_initializers().size() << "\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
