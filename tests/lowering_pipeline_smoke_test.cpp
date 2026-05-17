#include <string>

#include <gtest/gtest.h>

#include "codegen/lowering_pipeline.hpp"

#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

namespace
{

    void register_test_dialects (mlir::DialectRegistry& registry)
    {
        registry.insert<
            mlir::arith::ArithDialect,
            mlir::bufferization::BufferizationDialect,
            mlir::cf::ControlFlowDialect,
            mlir::func::FuncDialect,
            mlir::linalg::LinalgDialect,
            mlir::LLVM::LLVMDialect,
            mlir::memref::MemRefDialect,
            mlir::scf::SCFDialect,
            mlir::tensor::TensorDialect
        >();
    }

    mlir::OwningOpRef<mlir::ModuleOp> parse_module (
        mlir::MLIRContext& context,
        const char* source
    )
    {
        return mlir::parseSourceString<mlir::ModuleOp> (source, &context);
    }

    std::string print_module (mlir::ModuleOp module)
    {
        std::string output;
        llvm::raw_string_ostream os (output);
        module.print (os);
        os.flush ();
        return output;
    }

    TEST (LoweringPipelineSmoke, MinimalModulePasses)
    {
        mlir::DialectRegistry registry;
        register_test_dialects (registry);

        mlir::MLIRContext context (registry);
        context.loadAllAvailableDialects ();

        auto module = parse_module (context, R"mlir(
module {
  func.func @main() -> i32 {
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}
)mlir");

        ASSERT_TRUE (module);

        EXPECT_TRUE (mlir::succeeded (tc::run_lowering_pipeline (*module)));
    }

    TEST (LoweringPipelineSmoke, LinalgModuleLowersToLLVMDialect)
    {
        mlir::DialectRegistry registry;
        register_test_dialects (registry);

        mlir::MLIRContext context (registry);
        context.loadAllAvailableDialects ();

        auto module = parse_module (context, R"mlir(
module {
  func.func @main() {
    %value = arith.constant 1.0 : f32
    %alloc = memref.alloc() : memref<4xf32>
    linalg.fill ins(%value : f32) outs(%alloc : memref<4xf32>)
    memref.dealloc %alloc : memref<4xf32>
    return
  }
}
)mlir");

        ASSERT_TRUE (module);

        ASSERT_TRUE (mlir::succeeded (tc::run_lowering_pipeline (*module)));

        std::string lowered = print_module (*module);
        EXPECT_NE (lowered.find ("llvm.func"), std::string::npos);
        EXPECT_EQ (lowered.find ("linalg."), std::string::npos);
        EXPECT_EQ (lowered.find ("scf."), std::string::npos);
        EXPECT_EQ (lowered.find ("cf."), std::string::npos);
        EXPECT_EQ (lowered.find ("func.func"), std::string::npos);
    }

} // namespace
