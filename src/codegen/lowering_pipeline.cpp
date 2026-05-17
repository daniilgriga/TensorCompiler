#include "codegen/lowering_pipeline.hpp"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace tc
{
    mlir::LogicalResult run_lowering_pipeline (
        mlir::ModuleOp module,
        const LoweringPipelineOptions& options
    )
    {
        mlir::MLIRContext* ctx = module.getContext();

        mlir::DialectRegistry registry;
        mlir::arith::registerBufferizableOpInterfaceExternalModels (registry);
        mlir::linalg::registerBufferizableOpInterfaceExternalModels (registry);
        mlir::tensor::registerBufferizableOpInterfaceExternalModels (registry);
        mlir::scf::registerBufferizableOpInterfaceExternalModels (registry);
        mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels (registry);
        ctx->appendDialectRegistry (registry);

        mlir::PassManager pm (ctx);
        pm.enableVerifier (true);

        // --- cleanup before bufferization ---
        pm.addPass (mlir::createCanonicalizerPass ());
        pm.addPass (mlir::createCSEPass ());

        // --- bufferization: tensor (value semantics) → memref (memory semantics) ---
        pm.addPass (mlir::bufferization::createEmptyTensorToAllocTensorPass ());

        mlir::bufferization::OneShotBufferizePassOptions bufferize_options;
        bufferize_options.bufferizeFunctionBoundaries = true;

        pm.addPass (mlir::bufferization::createOneShotBufferizePass (bufferize_options));
        pm.addPass (mlir::createConvertBufferizationToMemRefPass ());
        pm.addPass (mlir::bufferization::createBufferDeallocationSimplificationPass ());
        pm.addPass (mlir::bufferization::createLowerDeallocationsPass ());
        pm.addPass (mlir::createCanonicalizerPass ());
        pm.addPass (mlir::createCSEPass ());

        // --- lower linalg: matmul/conv → explicit scf.for loops ---
        pm.addPass (mlir::createConvertLinalgToLoopsPass ());
        pm.addPass (mlir::createLowerAffinePass ());
        pm.addPass (mlir::createCanonicalizerPass ());
        pm.addPass (mlir::createCSEPass ());

        // --- lower control flow: scf.for → cf.br/cond_br ---
        pm.addPass (mlir::createSCFToControlFlowPass ());
        pm.addPass (mlir::memref::createExpandStridedMetadataPass ());
        pm.addPass (mlir::createCanonicalizerPass ());
        pm.addPass (mlir::createCSEPass ());

        mlir::ConvertFuncToLLVMPassOptions func_options;
        func_options.useBarePtrCallConv = options.use_bare_ptr_call_conv;

        // --- optional: generate _mlir_ciface_* wrappers for ExecutionEngine::invoke ---
        if (options.emit_c_interface)
            pm.nest<mlir::func::FuncOp> ().addPass (
                mlir::LLVM::createLLVMRequestCWrappersPass ());

        // --- lower remaining dialects to llvm dialect ---
        pm.addPass (mlir::createConvertFuncToLLVMPass (func_options));
        pm.addPass (mlir::createFinalizeMemRefToLLVMConversionPass ());
        pm.addPass (mlir::createConvertControlFlowToLLVMPass ());
        pm.addPass (mlir::createArithToLLVMConversionPass ());
        pm.addPass (mlir::createConvertIndexToLLVMPass ());

        // remove temporary type-conversion bridges left by dialect lowering passes
        pm.addPass (mlir::createReconcileUnrealizedCastsPass ());

        // --- final cleanup ---
        pm.addPass (mlir::createCanonicalizerPass ());
        pm.addPass (mlir::createCSEPass ());

        return pm.run (module);
    }

} // namespace tc
