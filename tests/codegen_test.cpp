#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "mlir/IR/Verifier.h"

#include "codegen/mlir_gen.hpp"
#include "graph/graph_builder.hpp"

#if __has_include(<sanitizer/lsan_interface.h>)
#include <sanitizer/lsan_interface.h>
#else
static void __lsan_disable() {}
static void __lsan_enable()  {}
#endif

namespace
{

// helper: create a float32 initializer with zero data
void make_initializer (tc::GraphBuilder& builder,
                       const std::string& name,
                       std::vector<int64_t> shape)
{
    tc::Value* v = builder.get_or_create_value (name);
    v->set_shape (shape);
    builder.mark_as_initializer (v);
    builder.set_value_dtype (v, tc::DType::Float32);

    int64_t count = 1;
    for (int64_t d : shape) count *= d;

    std::vector<uint8_t> data (count * sizeof(float), 0);
    builder.set_value_data (v, std::move (data));
}

// helper: create a graph input with float32 dtype
tc::Value* make_input (tc::GraphBuilder& builder,
                       const std::string& name,
                       std::vector<int64_t> shape)
{
    tc::Value* v = builder.get_or_create_value (name);
    v->set_shape (std::move (shape));
    builder.mark_as_input (v);
    builder.set_value_dtype (v, tc::DType::Float32);
    return v;
}

// helper: create a graph output
tc::Value* make_output (tc::GraphBuilder& builder,
                        const std::string& name)
{
    tc::Value* v = builder.get_or_create_value (name);
    builder.mark_as_output (v);
    return v;
}

TEST(CodegenTest, AddEmitsValidMLIR)
{
    tc::GraphBuilder builder;
    make_input  (builder, "A", {2, 3});
    make_input  (builder, "B", {2, 3});
    make_output (builder, "Y");

    builder.add_node ("Add",
        {builder.find_value ("A"), builder.find_value ("B")},
        {builder.find_value ("Y")});

    tc::MlirModule m = tc::graph_to_mlir (builder);
    EXPECT_TRUE (mlir::verify (*m.module).succeeded());
}

TEST(CodegenTest, MulEmitsValidMLIR)
{
    tc::GraphBuilder builder;
    make_input  (builder, "A", {4, 4});
    make_input  (builder, "B", {4, 4});
    make_output (builder, "Y");

    builder.add_node ("Mul",
        {builder.find_value ("A"), builder.find_value ("B")},
        {builder.find_value ("Y")});

    tc::MlirModule m = tc::graph_to_mlir (builder);
    EXPECT_TRUE (mlir::verify (*m.module).succeeded());
}

TEST(CodegenTest, ReluEmitsValidMLIR)
{
    tc::GraphBuilder builder;
    make_input  (builder, "X", {2, 3});
    make_output (builder, "Y");

    builder.add_node ("Relu",
        {builder.find_value ("X")},
        {builder.find_value ("Y")});

    tc::MlirModule m = tc::graph_to_mlir (builder);
    EXPECT_TRUE (mlir::verify (*m.module).succeeded());
}

TEST(CodegenTest, MatMulEmitsValidMLIR)
{
    tc::GraphBuilder builder;
    make_input  (builder, "A", {4, 8});
    make_input  (builder, "B", {8, 16});
    make_output (builder, "Y");

    builder.add_node ("MatMul",
        {builder.find_value ("A"), builder.find_value ("B")},
        {builder.find_value ("Y")});

    tc::MlirModule m = tc::graph_to_mlir (builder);
    EXPECT_TRUE (mlir::verify (*m.module).succeeded());
}

TEST(CodegenTest, GemmTransBWithBiasEmitsValidMLIR)
{
    tc::GraphBuilder builder;
    make_input       (builder, "A", {4, 8});
    make_initializer (builder, "W", {16, 8});
    make_initializer (builder, "B", {16});
    make_output      (builder, "Y");

    tc::Attributes attrs;
    attrs["transB"] = int64_t{1};
    attrs["alpha"]  = float{1.0f};
    attrs["beta"]   = float{1.0f};

    builder.add_node ("Gemm",
        {builder.find_value ("A"),
         builder.find_value ("W"),
         builder.find_value ("B")},
        {builder.find_value ("Y")},
        std::move (attrs));

    tc::MlirModule m = tc::graph_to_mlir (builder);
    EXPECT_TRUE (mlir::verify (*m.module).succeeded());
}

TEST(CodegenTest, GemmNoBiasEmitsValidMLIR)
{
    tc::GraphBuilder builder;
    make_input       (builder, "A", {4, 8});
    make_initializer (builder, "W", {8, 16});
    make_output      (builder, "Y");

    builder.add_node ("Gemm",
        {builder.find_value ("A"),
         builder.find_value ("W")},
        {builder.find_value ("Y")});

    tc::MlirModule m = tc::graph_to_mlir (builder);
    EXPECT_TRUE (mlir::verify (*m.module).succeeded());
}

TEST(CodegenTest, ConvNoPadEmitsValidMLIR)
{
    tc::GraphBuilder builder;
    make_input       (builder, "X", {1, 1, 5, 5});
    make_initializer (builder, "W", {1, 1, 3, 3});
    make_output      (builder, "Y");

    tc::Attributes attrs;
    attrs["kernel_shape"] = std::vector<int64_t>{3, 3};
    attrs["strides"]      = std::vector<int64_t>{1, 1};

    builder.add_node ("Conv",
        {builder.find_value ("X"),
         builder.find_value ("W")},
        {builder.find_value ("Y")},
        std::move (attrs));

    tc::MlirModule m = tc::graph_to_mlir (builder);
    EXPECT_TRUE (mlir::verify (*m.module).succeeded());
}

TEST(CodegenTest, ConvWithPadEmitsValidMLIR)
{
    tc::GraphBuilder builder;
    make_input       (builder, "X", {1, 1, 5, 5});
    make_initializer (builder, "W", {1, 1, 3, 3});
    make_output      (builder, "Y");

    tc::Attributes attrs;
    attrs["kernel_shape"] = std::vector<int64_t>{3, 3};
    attrs["strides"]      = std::vector<int64_t>{1, 1};
    attrs["pads"]         = std::vector<int64_t>{1, 1, 1, 1};

    builder.add_node ("Conv",
        {builder.find_value ("X"),
         builder.find_value ("W")},
        {builder.find_value ("Y")},
        std::move (attrs));

    tc::MlirModule m = tc::graph_to_mlir (builder);
    EXPECT_TRUE (mlir::verify (*m.module).succeeded());
}

TEST(CodegenTest, ConvWithBiasEmitsValidMLIR)
{
    tc::GraphBuilder builder;
    make_input       (builder, "X", {1, 3, 8, 8});
    make_initializer (builder, "W", {16, 3, 3, 3});
    make_initializer (builder, "B", {16});
    make_output      (builder, "Y");

    tc::Attributes attrs;
    attrs["kernel_shape"] = std::vector<int64_t>{3, 3};
    attrs["strides"]      = std::vector<int64_t>{1, 1};
    attrs["pads"]         = std::vector<int64_t>{1, 1, 1, 1};

    builder.add_node ("Conv",
        {builder.find_value ("X"),
         builder.find_value ("W"),
         builder.find_value ("B")},
        {builder.find_value ("Y")},
        std::move (attrs));

    tc::MlirModule m = tc::graph_to_mlir (builder);
    EXPECT_TRUE (mlir::verify (*m.module).succeeded());
}

TEST(CodegenTest, ReshapeEmitsValidMLIR)
{
    tc::GraphBuilder builder;
    make_input       (builder, "X", {1, 3, 3});
    make_initializer (builder, "shape", {2});

    // set int64 shape data: [1, 9]
    tc::Value* sv = builder.find_value ("shape");
    sv->set_shape ({2});
    builder.set_value_dtype (sv, tc::DType::Int64);
    std::vector<int64_t> shape_vals = {1, 9};
    std::vector<uint8_t> raw (sizeof (int64_t) * 2);
    std::memcpy (raw.data(), shape_vals.data(), raw.size());
    builder.set_value_data (sv, std::move (raw));

    make_output (builder, "Y");

    builder.add_node ("Reshape",
        {builder.find_value ("X"),
         builder.find_value ("shape")},
        {builder.find_value ("Y")});

    tc::MlirModule m = tc::graph_to_mlir (builder);
    EXPECT_TRUE (mlir::verify (*m.module).succeeded());
}

TEST(CodegenTest, UnsupportedOpThrows)
{
    tc::GraphBuilder builder;
    make_input  (builder, "X", {2, 3});
    make_output (builder, "Y");

    builder.add_node ("FakeOp",
        {builder.find_value ("X")},
        {builder.find_value ("Y")});

    // MLIR ops leak when exception interrupts emit - suppress LSAN for this test
    __lsan_disable();
    EXPECT_THROW (tc::graph_to_mlir (builder), std::runtime_error);
    __lsan_enable();
}

} // namespace
