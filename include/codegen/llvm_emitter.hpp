#pragma once

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/CodeGen.h"

namespace tc
{
    enum class OutputKind
    {
        LLVM_IR,  // --emit-llvm
        ASM,      // --emit-asm
        OBJ,      // --emit-obj
        SO        // --emit-so (shared library, requires PIC)
    };

    struct EmitOptions
    {
        std::string target_triple;                                    // "" = native
        std::string cpu;                                              // "" = "generic"
        std::string features;                                         // "" = no extra features
        std::string output_path;                                      // "" = stdout (or "a.o" for OBJ)
        llvm::CodeGenOptLevel opt_level = llvm::CodeGenOptLevel::None; // -O0 default
        OutputKind output_kind = OutputKind::ASM;
    };

    struct RunOptions
    {
        std::string input_path;
        int64_t N = 1, C = 1, H = 1, W = 1;
        int64_t out_elems = 0;
    };

    int emit_output (mlir::ModuleOp module, const EmitOptions& options);
    int run_jit    (mlir::ModuleOp module, const RunOptions& options);

} // namespace tc
