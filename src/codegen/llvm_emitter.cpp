#include "codegen/llvm_emitter.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"

#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

namespace tc
{
namespace
{
    void initialize_codegen_backends ()
    {
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmPrinters();
        llvm::InitializeAllAsmParsers();
    }

    std::unique_ptr<llvm::Module> translate_to_llvm_ir (
        mlir::ModuleOp module,
        llvm::LLVMContext& llvm_context
    )
    {
        mlir::registerBuiltinDialectTranslation (*module.getContext());
        mlir::registerLLVMDialectTranslation (*module.getContext());

        auto llvm_module = mlir::translateModuleToLLVMIR (module, llvm_context);

        if (!llvm_module)
        {
            llvm::errs() << "error: failed to translate MLIR to LLVM IR\n";
            return nullptr;
        }

        return llvm_module;
    }

    bool open_file_stream (
        const std::string& path,
        std::unique_ptr<llvm::raw_fd_ostream>& file_stream
    )
    {
        std::error_code ec;
        file_stream = std::make_unique<llvm::raw_fd_ostream> (
            path, ec, llvm::sys::fs::OF_None);
        if (ec)
        {
            llvm::errs() << "error: cannot open output file '" << path
                         << "': " << ec.message() << "\n";
            return false;
        }
        return true;
    }

    int emit_llvm_ir (llvm::Module& llvm_module, const EmitOptions& options)
    {
        if (options.output_kind == OutputKind::LLVM_IR)
        {
            if (!options.output_path.empty())
            {
                std::unique_ptr<llvm::raw_fd_ostream> file_stream;
                if (!open_file_stream (options.output_path, file_stream))
                    return 1;

                llvm_module.print (*file_stream, nullptr);
                file_stream->flush();
                return 0;
            }

            llvm_module.print (llvm::outs(), nullptr);
            return 0;
        }

        llvm::errs() << "error: invalid output kind for emit_llvm_ir\n";
        return 1;
    }

    std::unique_ptr<llvm::TargetMachine> create_target_machine (
        llvm::Module& llvm_module,
        const EmitOptions& options
    )
    {
        std::string triple = options.target_triple.empty()
            ? llvm::sys::getDefaultTargetTriple()
            : options.target_triple;

        llvm::Triple target_triple (triple);
        llvm_module.setTargetTriple (target_triple);

        std::string error_msg;
        const llvm::Target* target =
            llvm::TargetRegistry::lookupTarget (target_triple, error_msg);

        if (!target)
        {
            llvm::errs() << "error: unknown target '" << triple
                          << "': " << error_msg << "\n";
            return nullptr;
        }

        std::string cpu = options.cpu.empty() ? "generic" : options.cpu;

        llvm::TargetOptions target_opts;
        auto reloc = (options.output_kind == OutputKind::SO)
            ? std::optional<llvm::Reloc::Model> (llvm::Reloc::PIC_)
            : std::nullopt;

        auto target_machine = std::unique_ptr<llvm::TargetMachine> (
            target->createTargetMachine (
                target_triple,
                cpu,
                options.features,
                target_opts,
                reloc,
                std::nullopt,
                options.opt_level
            )
        );

        if (!target_machine)
        {
            llvm::errs() << "error: failed to create target machine\n";
            return nullptr;
        }

        llvm_module.setDataLayout (target_machine->createDataLayout());
        return target_machine;
    }

    llvm::OptimizationLevel to_optimization_level (llvm::CodeGenOptLevel opt_level)
    {
        switch (opt_level)
        {
            case llvm::CodeGenOptLevel::None:       return llvm::OptimizationLevel::O0;
            case llvm::CodeGenOptLevel::Less:       return llvm::OptimizationLevel::O1;
            case llvm::CodeGenOptLevel::Default:    return llvm::OptimizationLevel::O2;
            case llvm::CodeGenOptLevel::Aggressive: return llvm::OptimizationLevel::O3;
        }
        return llvm::OptimizationLevel::O0;
    }

    void run_llvm_optimization_pipeline (
        llvm::Module& llvm_module,
        llvm::TargetMachine& target_machine,
        llvm::CodeGenOptLevel opt_level
    )
    {
        if (opt_level == llvm::CodeGenOptLevel::None)
            return;

        llvm::PassBuilder pass_builder (&target_machine);

        llvm::LoopAnalysisManager loop_am;
        llvm::FunctionAnalysisManager function_am;
        llvm::CGSCCAnalysisManager cgscc_am;
        llvm::ModuleAnalysisManager module_am;

        pass_builder.registerModuleAnalyses (module_am);
        pass_builder.registerCGSCCAnalyses (cgscc_am);
        pass_builder.registerFunctionAnalyses (function_am);
        pass_builder.registerLoopAnalyses (loop_am);
        pass_builder.crossRegisterProxies (loop_am, function_am, cgscc_am, module_am);

        llvm::ModulePassManager module_pm =
            pass_builder.buildPerModuleDefaultPipeline (to_optimization_level (opt_level));
        module_pm.run (llvm_module, module_am);
    }

    int emit_codegen_file (
        llvm::Module& llvm_module,
        llvm::TargetMachine& target_machine,
        const EmitOptions& options
    )
    {
        llvm::legacy::PassManager pass_manager;

        bool is_obj = (options.output_kind == OutputKind::OBJ ||
                       options.output_kind == OutputKind::SO);
        auto file_type = is_obj
            ? llvm::CodeGenFileType::ObjectFile
            : llvm::CodeGenFileType::AssemblyFile;

        std::string output_path = options.output_path;
        if (options.output_kind == OutputKind::OBJ && output_path.empty())
            output_path = "a.o";
        if (options.output_kind == OutputKind::SO && output_path.empty())
            output_path = "a.so";

        llvm::raw_pwrite_stream* output_stream = &llvm::outs();
        std::unique_ptr<llvm::raw_fd_ostream> file_stream;
        if (!output_path.empty())
        {
            if (!open_file_stream (output_path, file_stream))
                return 1;
            output_stream = file_stream.get();
        }

        if (target_machine.addPassesToEmitFile (
                pass_manager, *output_stream, nullptr, file_type))
        {
            llvm::errs () << "error: cannot emit file for this target\n";
            return 1;
        }

        pass_manager.run (llvm_module);
        if (file_stream)
            file_stream->flush();

        if (options.output_kind == OutputKind::SO)
        {
            // link the object file into a shared library
            std::string tmp_obj = output_path + ".tmp.o";
            std::rename (output_path.c_str(), tmp_obj.c_str());

#if defined(__APPLE__)
            std::string link_cmd = "clang -shared -o " + output_path + " " + tmp_obj;
#else
            std::string link_cmd = "clang -shared -fPIC -o " + output_path + " " + tmp_obj;
#endif
            int rc = std::system (link_cmd.c_str());
            std::remove (tmp_obj.c_str());
            if (rc != 0)
            {
                llvm::errs() << "error: linking shared library failed\n";
                return 1;
            }
        }

        return 0;
    }

} // namespace

    int emit_output (mlir::ModuleOp module, const EmitOptions& options)
    {
        initialize_codegen_backends ();

        llvm::LLVMContext llvm_context;
        auto llvm_module = translate_to_llvm_ir (module, llvm_context);
        if (!llvm_module)
            return 1;

        std::unique_ptr<llvm::TargetMachine> target_machine;
        if (options.opt_level != llvm::CodeGenOptLevel::None ||
            options.output_kind != OutputKind::LLVM_IR)
        {
            target_machine = create_target_machine (*llvm_module, options);
            if (!target_machine)
                return 1;
        }

        if (target_machine)
            run_llvm_optimization_pipeline (
                *llvm_module, *target_machine, options.opt_level);

        if (options.output_kind == OutputKind::LLVM_IR)
            return emit_llvm_ir (*llvm_module, options);

        return emit_codegen_file (*llvm_module, *target_machine, options);
    }

    int run_jit (mlir::ModuleOp module, const RunOptions& options)
    {
        initialize_codegen_backends ();

        // register dialect translations required by ExecutionEngine
        mlir::registerBuiltinDialectTranslation (*module.getContext());
        mlir::registerLLVMDialectTranslation (*module.getContext());

        mlir::ExecutionEngineOptions eng_opts;
        eng_opts.transformer = mlir::makeOptimizingTransformer (0, 0, nullptr);
        eng_opts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Default;

        auto maybe_engine = mlir::ExecutionEngine::create (module, eng_opts);
        if (!maybe_engine)
        {
            llvm::errs() << "error: failed to create JIT engine: "
                         << llvm::toString (maybe_engine.takeError()) << "\n";
            return 1;
        }
        auto& engine = *maybe_engine;

        // read input from binary file
        const int64_t in_elems = options.N * options.C * options.H * options.W;
        std::vector<float> input (static_cast<std::size_t> (in_elems));

        if (!options.input_path.empty())
        {
            FILE* f = std::fopen (options.input_path.c_str(), "rb");
            if (!f)
            {
                llvm::errs() << "error: cannot open input file '"
                             << options.input_path << "'\n";
                return 1;
            }
            if (static_cast<int64_t> (
                    std::fread (input.data(), sizeof(float), in_elems, f)) != in_elems)
            {
                llvm::errs() << "error: expected " << in_elems
                             << " floats in input file\n";
                std::fclose (f);
                return 1;
            }
            std::fclose (f);
        }

        // build input memref descriptor (rank-4, NCHW)
        StridedMemRefType<float, 4> in_desc;
        in_desc.basePtr = input.data();
        in_desc.data    = input.data();
        in_desc.offset  = 0;
        in_desc.sizes[0] = options.N;
        in_desc.sizes[1] = options.C;
        in_desc.sizes[2] = options.H;
        in_desc.sizes[3] = options.W;
        in_desc.strides[0] = options.C * options.H * options.W;
        in_desc.strides[1] = options.H * options.W;
        in_desc.strides[2] = options.W;
        in_desc.strides[3] = 1;

        // build output memref descriptor (rank-2, [1 x out_elems])
        const int64_t out_elems = options.out_elems > 0 ? options.out_elems : 1;
        std::vector<float> output (static_cast<std::size_t> (out_elems), 0.0f);

        StridedMemRefType<float, 2> out_desc;
        out_desc.basePtr  = output.data();
        out_desc.data     = output.data();
        out_desc.offset   = 0;
        out_desc.sizes[0] = 1;
        out_desc.sizes[1] = out_elems;
        out_desc.strides[0] = out_elems;
        out_desc.strides[1] = 1;

        // _mlir_ciface_main(void* out_descriptor, void* in_descriptor)
        // Direct call with correct ABI — invokePacked can't handle (ptr, ptr) signature
        auto sym = engine->lookup ("_mlir_ciface_main");
        if (!sym)
        {
            llvm::errs() << "error: cannot find '_mlir_ciface_main': "
                         << llvm::toString (sym.takeError()) << "\n";
            return 1;
        }

        using CIfaceMain = void (*)(void*, void*);
        auto fn = reinterpret_cast<CIfaceMain> (*sym);
        fn (&out_desc, &in_desc);

        // print output as space-separated floats
        for (int64_t i = 0; i < out_elems; ++i)
            llvm::outs() << out_desc.data[i]
                         << (i + 1 < out_elems ? " " : "\n");

        return 0;
    }

} // namespace tc
