#include "codegen/llvm_emitter.hpp"

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
        auto target_machine = std::unique_ptr<llvm::TargetMachine> (
            target->createTargetMachine (
                target_triple,
                cpu,
                options.features,
                target_opts,
                std::nullopt,
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

        auto file_type = (options.output_kind == OutputKind::OBJ)
            ? llvm::CodeGenFileType::ObjectFile
            : llvm::CodeGenFileType::AssemblyFile;

        std::string output_path = options.output_path;
        if (options.output_kind == OutputKind::OBJ && output_path.empty())
            output_path = "a.o";

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

} // namespace tc
