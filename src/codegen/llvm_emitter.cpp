#include "codegen/llvm_emitter.hpp"

#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

namespace tc
{
    int emit_output (mlir::ModuleOp module, const EmitOptions& options)
    {
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmPrinters();
        llvm::InitializeAllAsmParsers();

        mlir::registerBuiltinDialectTranslation (*module.getContext());
        mlir::registerLLVMDialectTranslation (*module.getContext());

        llvm::LLVMContext llvm_context;
        auto llvm_module = mlir::translateModuleToLLVMIR (module, llvm_context);

        if (!llvm_module)
        {
            llvm::errs() << "error: failed to translate MLIR to LLVM IR\n";
            return 1;
        }

        if (options.output_kind == OutputKind::LLVM_IR)
        {
            llvm_module->print (llvm::outs(), nullptr);
            return 0;
        }

        std::string triple = options.target_triple.empty()
            ? llvm::sys::getDefaultTargetTriple()
            : options.target_triple;

        llvm_module->setTargetTriple (llvm::Triple (triple));

        std::string error_msg;
        const llvm::Target* target =
            llvm::TargetRegistry::lookupTarget (llvm::Triple (triple), error_msg);

        if (!target)
        {
            llvm::errs() << "error: unknown target '" << triple
                          << "': " << error_msg << "\n";
            return 1;
        }

        std::string cpu = options.cpu.empty() ? "generic" : options.cpu;

        llvm::TargetOptions target_opts;
        auto target_machine = std::unique_ptr<llvm::TargetMachine> (
            target->createTargetMachine (
                llvm::Triple (triple),
                cpu,
                options.features,
                target_opts,
                std::nullopt,
                std::nullopt,
                options.opt_level
            )
        );

        llvm_module->setDataLayout (target_machine->createDataLayout());

        llvm::legacy::PassManager pass_manager;

        auto file_type = (options.output_kind == OutputKind::OBJ)
            ? llvm::CodeGenFileType::ObjectFile
            : llvm::CodeGenFileType::AssemblyFile;

        if (target_machine->addPassesToEmitFile (
                pass_manager, llvm::outs(), nullptr, file_type))
        {
            llvm::errs () << "error: cannot emit file for this target\n";
            return 1;
        }

        pass_manager.run (*llvm_module);
        return 0;
    }

} // namespace tc