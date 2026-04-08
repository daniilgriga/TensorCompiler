#include "codegen/lowering_pipeline.hpp"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/Passes.h"

namespace tc
{
    mlir::LogicalResult run_lowering_pipeline (
        mlir::ModuleOp module,
        const LoweringPipelineOptions& options
    )
    {
        mlir::PassManager pm (module.getContext());

        pm.enableVerifier (options.enable_verifier);

        pm.addPass (mlir::createCanonicalizerPass ());
        pm.addPass (mlir::createCSEPass ());

        mlir::bufferization::OneShotBufferizePassOptions bufferize_options;
        bufferize_options.bufferizeFunctionBoundaries = options.bufferize_function_boundaries;

        pm.addPass (mlir::bufferization::createOneShotBufferizePass (bufferize_options));
        pm.addPass (mlir::createCanonicalizerPass ());
        pm.addPass (mlir::createCSEPass ());

        pm.addPass (mlir::createConvertLinalgToLoopsPass ());
        pm.addPass (mlir::createCanonicalizerPass ());
        pm.addPass (mlir::createCSEPass ());

        pm.addPass (mlir::createSCFToControlFlowPass ());
        pm.addPass (mlir::createCanonicalizerPass ());
        pm.addPass (mlir::createCSEPass ());

        mlir::ArithToLLVMConversionPassOptions arith_options;
        arith_options.indexBitwidth = options.index_bitwidth;

        mlir::ConvertIndexToLLVMPassOptions index_options;
        index_options.indexBitwidth = options.index_bitwidth;

        mlir::FinalizeMemRefToLLVMConversionPassOptions memref_options;
        memref_options.indexBitwidth = options.index_bitwidth;
        memref_options.useAlignedAlloc = options.use_aligned_alloc;

        mlir::ConvertFuncToLLVMPassOptions func_options;
        func_options.indexBitwidth = options.index_bitwidth;
        func_options.useBarePtrCallConv = options.use_bare_ptr_call_conv;

        pm.addPass (mlir::createArithToLLVMConversionPass (arith_options));
        pm.addPass (mlir::createConvertIndexToLLVMPass (index_options));
        pm.addPass (mlir::createConvertControlFlowToLLVMPass ());
        pm.addPass (mlir::createFinalizeMemRefToLLVMConversionPass (memref_options));
        pm.addPass (mlir::createConvertFuncToLLVMPass (func_options));

        if (!options.llvm_data_layout.empty ())
        {
            mlir::SetLLVMModuleDataLayoutPassOptions data_layout_options;
            data_layout_options.dataLayout = options.llvm_data_layout;
            pm.addPass (mlir::createSetLLVMModuleDataLayoutPass (data_layout_options));
        }

        pm.addPass (mlir::createCanonicalizerPass ());
        pm.addPass (mlir::createCSEPass ());

        pm.addPass (mlir::createReconcileUnrealizedCastsPass ());

        return pm.run (module);
    }
} // namespace tc
