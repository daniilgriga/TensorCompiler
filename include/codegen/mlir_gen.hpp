#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include "graph/graph_builder.hpp"

namespace tc
{
    // translate a fully-imported GraphBuilder IR into an MLIR module
    mlir::OwningOpRef<mlir::ModuleOp> graph_to_mlir (const GraphBuilder& builder);

} // namespace tc
