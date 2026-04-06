#pragma once

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace tc
{
    struct LoweringPipelineOptions
    {
        bool enable_verifier = true;
        bool bufferize_function_boundaries = true;
        unsigned index_bitwidth = 0;
        bool use_bare_ptr_call_conv = false;
        bool use_aligned_alloc = false;
        std::string llvm_data_layout;
    };

    mlir::LogicalResult run_lowering_pipeline (
        mlir::ModuleOp module,
        const LoweringPipelineOptions& options = {}
    );
} // namespace tc
