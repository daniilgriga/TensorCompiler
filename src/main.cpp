#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "importer/importer.hpp"
#include "graph/dot_export.hpp"

#ifdef TC_WITH_CODEGEN
#include "codegen/mlir_gen.hpp"
#include "codegen/lowering_pipeline.hpp"
#include "codegen/llvm_emitter.hpp"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#endif

// CLI options
#ifdef TC_WITH_CODEGEN

static llvm::cl::opt<std::string> input_file (
    llvm::cl::Positional,
    llvm::cl::desc ("<model.onnx>"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> dot_file (
    "dot",
    llvm::cl::desc ("Export graph to Graphviz dot file"),
    llvm::cl::value_desc ("file"),
    llvm::cl::init (""));

static llvm::cl::opt<bool> emit_mlir (
    "emit-mlir",
    llvm::cl::desc ("Print MLIR (Linalg dialect) to stdout"));

static llvm::cl::opt<bool> emit_llvm (
    "emit-llvm",
    llvm::cl::desc ("Print LLVM IR to stdout"));

static llvm::cl::opt<bool> emit_asm (
    "emit-asm",
    llvm::cl::desc ("Print assembly to stdout"));

static llvm::cl::opt<bool> emit_obj (
    "emit-obj",
    llvm::cl::desc ("Emit object file"));

static llvm::cl::opt<std::string> output_file (
    "o",
    llvm::cl::desc ("Output file for --emit-llvm/--emit-asm/--emit-obj"),
    llvm::cl::value_desc ("file"),
    llvm::cl::init (""));

static llvm::cl::opt<char> opt_level (
    "O",
    llvm::cl::desc ("Optimization level (0/1/2/3)"),
    llvm::cl::Prefix,
    llvm::cl::init ('0'));

static llvm::cl::opt<std::string> target_triple (
    "target",
    llvm::cl::desc ("Target triple (default: native)"),
    llvm::cl::value_desc ("triple"),
    llvm::cl::init (""));

#endif // TC_WITH_CODEGEN

int main (int argc, char* argv[])
{
#ifdef TC_WITH_CODEGEN

    llvm::cl::ParseCommandLineOptions (argc, argv, "TensorCompiler — ONNX -> Assembly\n");

    try
    {
        tc::GraphBuilder builder = tc::import_onnx (input_file);

        // --dot <file>
        if (!dot_file.empty())
        {
            std::ofstream ofs (dot_file);
            if (!ofs)
            {
                std::cerr << "Error: cannot open " << dot_file << "\n";
                return 1;
            }
            tc::dump_dot (builder, ofs);
            std::cout << "dot exported to " << dot_file << "\n";
            return 0;
        }

        // --emit-mlir
        if (emit_mlir)
        {
            tc::MlirModule m = tc::graph_to_mlir (builder);
            m.module->print (llvm::outs());
            llvm::outs() << "\n";
            return 0;
        }

        const int emit_modes =
            static_cast<int> (emit_llvm) +
            static_cast<int> (emit_asm) +
            static_cast<int> (emit_obj);

        if (emit_modes > 1)
        {
            std::cerr << "Error: choose only one of --emit-llvm, --emit-asm, --emit-obj\n";
            return 1;
        }

        // --emit-llvm / --emit-asm / --emit-obj
        if (emit_modes == 1)
        {
            tc::MlirModule m = tc::graph_to_mlir (builder);

            if (mlir::failed (tc::run_lowering_pipeline (*m.module)))
            {
                std::cerr << "Error: lowering pipeline failed\n";
                return 1;
            }

            tc::EmitOptions opts;

            if (emit_llvm)      opts.output_kind = tc::OutputKind::LLVM_IR;
            else if (emit_asm)  opts.output_kind = tc::OutputKind::ASM;
            else                opts.output_kind = tc::OutputKind::OBJ;

            opts.target_triple = target_triple.getValue();
            opts.output_path = output_file.getValue();

            switch (opt_level)
            {
                case '1': opts.opt_level = llvm::CodeGenOptLevel::Less;       break;
                case '2': opts.opt_level = llvm::CodeGenOptLevel::Default;    break;
                case '3': opts.opt_level = llvm::CodeGenOptLevel::Aggressive; break;
                default:  opts.opt_level = llvm::CodeGenOptLevel::None;       break;
            }

            return tc::emit_output (*m.module, opts);
        }

        // default: print stats
        std::cout << "nodes:        " << builder.nodes().size()             << "\n"
                  << "values:       " << builder.values().size()             << "\n"
                  << "inputs:       " << builder.graph_inputs().size()       << "\n"
                  << "outputs:      " << builder.graph_outputs().size()      << "\n"
                  << "initializers: " << builder.graph_initializers().size() << "\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

#else

    if (argc < 2)
    {
        std::cerr << "Usage:\n"
                  << "  tc_main <model.onnx>               print graph stats\n"
                  << "  tc_main <model.onnx> --dot <file>  export graph to Graphviz dot\n";
        return 1;
    }

    try
    {
        tc::GraphBuilder builder = tc::import_onnx (argv[1]);

        if (argc >= 4 && std::strcmp (argv[2], "--dot") == 0)
        {
            std::ofstream ofs (argv[3]);
            if (!ofs)
            {
                std::cerr << "Error: cannot open " << argv[3] << "\n";
                return 1;
            }
            tc::dump_dot (builder, ofs);
            std::cout << "dot exported to " << argv[3] << "\n";
            return 0;
        }

        std::cout << "nodes:        " << builder.nodes().size()             << "\n"
                  << "values:       " << builder.values().size()             << "\n"
                  << "inputs:       " << builder.graph_inputs().size()       << "\n"
                  << "outputs:      " << builder.graph_outputs().size()      << "\n"
                  << "initializers: " << builder.graph_initializers().size() << "\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

#endif // TC_WITH_CODEGEN

    return 0;
}
