#include "codegen/lowering_pipeline.hpp"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"



namespace tc
{
    mlir::LogicalResult run_lowering_pipeline (mlir::ModuleOp module)
    {
        mlir::PassManager pm (module.getContext());

        pm.enableVerifier (true);

        pm.addPass (mlir::createCanonicalizerPass());
        pm.addPass (mlir::createCSEPass ());

        return pm.run (module);
    }
} // namespace tc