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

        pm.enableVerifier (options.enable_verifier);

        pm.addPass (mlir::createCanonicalizerPass ());
        pm.addPass (mlir::createCSEPass ());

        pm.addPass (mlir::bufferization::createEmptyTensorToAllocTensorPass ());

        mlir::bufferization::OneShotBufferizePassOptions bufferize_options;
        bufferize_options.bufferizeFunctionBoundaries = options.bufferize_function_boundaries;

        pm.addPass (mlir::bufferization::createOneShotBufferizePass (bufferize_options));

        pm.addPass (mlir::createConvertBufferizationToMemRefPass ());

        pm.addPass (mlir::bufferization::createBufferDeallocationSimplificationPass ());
        pm.addPass (mlir::bufferization::createLowerDeallocationsPass ());
        pm.addPass (mlir::createCanonicalizerPass ());
        pm.addPass (mlir::createCSEPass ());

        pm.addPass (mlir::createConvertLinalgToLoopsPass ());
        pm.addPass (mlir::createLowerAffinePass ());
        pm.addPass (mlir::createCanonicalizerPass ());
        pm.addPass (mlir::createCSEPass ());

        pm.addPass (mlir::createSCFToControlFlowPass ());
        pm.addPass (mlir::memref::createExpandStridedMetadataPass ());
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

        if (options.emit_c_interface)
            pm.nest<mlir::func::FuncOp> ().addPass (
                mlir::LLVM::createLLVMRequestCWrappersPass ());

        pm.addPass (mlir::createConvertFuncToLLVMPass (func_options));
        pm.addPass (mlir::createFinalizeMemRefToLLVMConversionPass (memref_options));
        pm.addPass (mlir::createConvertControlFlowToLLVMPass ());
        pm.addPass (mlir::createConvertMathToLLVMPass ());
        pm.addPass (mlir::createArithToLLVMConversionPass (arith_options));
        pm.addPass (mlir::createConvertIndexToLLVMPass (index_options));

        if (!options.llvm_data_layout.empty ())
        {
            mlir::SetLLVMModuleDataLayoutPassOptions data_layout_options;
            data_layout_options.dataLayout = options.llvm_data_layout;
            pm.addPass (mlir::createSetLLVMModuleDataLayoutPass (data_layout_options));
        }

        pm.addPass (mlir::createReconcileUnrealizedCastsPass ());

        pm.addPass (mlir::createCanonicalizerPass ());
        pm.addPass (mlir::createCSEPass ());

        return pm.run (module);
    }

} // namespace tc
