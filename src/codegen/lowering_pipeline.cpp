#include "codegen/lowering_pipeline.hpp"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/Passes.h"

namespace tc
{
    mlir::LogicalResult run_lowering_pipeline (mlir::ModuleOp module)
    {
        mlir::PassManager pm (module.getContext());

        pm.enableVerifier (true);

        pm.addPass (mlir::createCanonicalizerPass());
        pm.addPass (mlir::createCSEPass());

        mlir::bufferization::OneShotBufferizePassOptions bufferize_options;
        bufferize_options.bufferizeFunctionBoundaries = true;

        pm.addPass (mlir::bufferization::createOneShotBufferizePass (bufferize_options));
        pm.addPass (mlir::createCanonicalizerPass());
        pm.addPass (mlir::createCSEPass());

        pm.addPass (mlir::createConvertLinalgToLoopsPass());
        pm.addPass (mlir::createCanonicalizerPass());
        pm.addPass (mlir::createCSEPass());

        pm.addPass (mlir::createSCFToControlFlowPass());
        pm.addPass (mlir::createCanonicalizerPass());
        pm.addPass (mlir::createCSEPass());

        pm.addPass (mlir::createArithToLLVMConversionPass());
        pm.addPass (mlir::createConvertIndexToLLVMPass());
        pm.addPass (mlir::createConvertControlFlowToLLVMPass());
        pm.addPass (mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass (mlir::createConvertFuncToLLVMPass());

        pm.addPass (mlir::createCanonicalizerPass());
        pm.addPass (mlir::createCSEPass ());

        pm.addPass (mlir::createReconcileUnrealizedCastsPass());

        return pm.run (module);
    }
} // namespace tc