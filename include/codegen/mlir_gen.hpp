#pragma once

#include <memory>
#include <utility>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "graph/graph_builder.hpp"

namespace tc
{
    // translate a fully-imported GraphBuilder IR into an MLIR module
    struct MlirModule
    {
        std::unique_ptr<mlir::MLIRContext> context;
        mlir::OwningOpRef<mlir::ModuleOp> module;
    };

    MlirModule graph_to_mlir (const GraphBuilder& builder);

} // namespace tc
