#pragma once

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace tc
{
    struct LoweringPipelineOptions
    {
        // bare pointer ABI: func takes float* instead of full memref descriptor.
        bool use_bare_ptr_call_conv = false;

        // emit _mlir_ciface_* C-interface wrappers required by ExecutionEngine::invoke
        bool emit_c_interface = false;
    };

    mlir::LogicalResult run_lowering_pipeline (
        mlir::ModuleOp module,
        const LoweringPipelineOptions& options = {}
    );
} // namespace tc
