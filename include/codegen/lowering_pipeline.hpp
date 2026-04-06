#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace tc
{
    mlir::LogicalResult run_lowering_pipeline (mlir::ModuleOp module);
} // namespace tc