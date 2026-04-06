#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "codegen/mlir_gen.hpp"
#include "graph/graph_builder.hpp"
#include "graph/node.hpp"
#include "graph/value.hpp"

namespace tc
{

namespace
{

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// map tc::DType to the corresponding MLIR element type
mlir::Type dtype_to_mlir (DType dtype, mlir::MLIRContext* ctx)
{
    switch (dtype)
    {
        case DType::Float32:  return mlir::Float32Type::get (ctx);
        case DType::Float64:  return mlir::Float64Type::get (ctx);
        case DType::Float16:  return mlir::Float16Type::get (ctx);
        case DType::BFloat16: return mlir::BFloat16Type::get (ctx);
        case DType::Int8:     return mlir::IntegerType::get (ctx, 8);
        case DType::Int16:    return mlir::IntegerType::get (ctx, 16);
        case DType::Int32:    return mlir::IntegerType::get (ctx, 32);
        case DType::Int64:    return mlir::IntegerType::get (ctx, 64);
        default:
            throw std::runtime_error (
                "dtype_to_mlir: unsupported DType " +
                std::to_string (static_cast<int32_t> (dtype)));
    }
}

// build a RankedTensorType from a Value's shape and dtype
mlir::RankedTensorType value_to_tensor_type (const Value& val, mlir::MLIRContext* ctx)
{
    mlir::Type elem = dtype_to_mlir (val.dtype(), ctx);

    llvm::SmallVector<int64_t> dims;
    dims.reserve (val.shape().size());
    for (int64_t d : val.shape())
        dims.push_back (d > 0 ? d : mlir::ShapedType::kDynamic);

    return mlir::RankedTensorType::get (dims, elem);
}

// ---------------------------------------------------------------------------
// MLIRCodegen
// ---------------------------------------------------------------------------

class MLIRCodegen
{
public:
    explicit MLIRCodegen (const GraphBuilder& builder)
        : builder_ (builder)
    {
        // register all dialects we emit
        ctx_.loadDialect<mlir::func::FuncDialect,
                         mlir::arith::ArithDialect,
                         mlir::linalg::LinalgDialect,
                         mlir::tensor::TensorDialect>();
    }

    mlir::OwningOpRef<mlir::ModuleOp> emit ()
    {
        auto loc = mlir::UnknownLoc::get (&ctx_);

        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::ModuleOp::create (loc);

        mlir::OpBuilder b (module->getBodyRegion());

        emit_func (b, loc, *module);

        return module;
    }

private:
    const GraphBuilder& builder_;
    mlir::MLIRContext   ctx_;

    // SSA values for graph values by name
    std::unordered_map<std::string, mlir::Value> value_map_;

    // -----------------------------------------------------------------------
    // func.func emission
    // -----------------------------------------------------------------------

    void emit_func (mlir::OpBuilder& b,
                    mlir::Location   loc,
                    mlir::ModuleOp   module)
    {
        // build function signature: (graph_inputs...) -> (graph_outputs...)
        llvm::SmallVector<mlir::Type> arg_types;
        for (const Value* v : builder_.graph_inputs())
            arg_types.push_back (value_to_tensor_type (*v, &ctx_));

        llvm::SmallVector<mlir::Type> res_types;
        for (const Value* v : builder_.graph_outputs())
            res_types.push_back (value_to_tensor_type (*v, &ctx_));

        auto func_type = b.getFunctionType (arg_types, res_types);
        auto func_op   = mlir::func::FuncOp::create (loc, "main", func_type);

        mlir::Block* entry = func_op.addEntryBlock();
        mlir::OpBuilder body_builder (entry, entry->end());

        // bind block arguments to graph input names
        const auto& inputs = builder_.graph_inputs();
        for (std::size_t i = 0; i < inputs.size(); ++i)
            value_map_[inputs[i]->name()] = entry->getArgument (i);

        // emit constants for initializers
        emit_initializers (body_builder, loc);

        // emit ops in topological order (ONNX spec guarantees it)
        for (const auto& node : builder_.nodes())
            emit_node (body_builder, loc, *node);

        // collect outputs and emit return
        llvm::SmallVector<mlir::Value> results;
        for (const Value* v : builder_.graph_outputs())
        {
            auto it = value_map_.find (v->name());
            if (it == value_map_.end())
                throw std::runtime_error (
                    "emit_func: output value not found: " + v->name());
            results.push_back (it->second);
        }

        mlir::func::ReturnOp::create (body_builder, loc, results);

        module.push_back (func_op);
    }

    void emit_initializers (mlir::OpBuilder& b, mlir::Location loc)
    {
        for (const Value* v : builder_.graph_initializers())
        {
            if (value_map_.count (v->name()))
                continue; // already emitted

            mlir::RankedTensorType tensor_type = value_to_tensor_type (*v, &ctx_);

            // build a dense elements attribute from raw bytes
            if (v->data().empty())
                throw std::runtime_error (
                    "emit_initializers: no data for initializer '" + v->name() + "'");

            mlir::DenseElementsAttr dense_attr =
                mlir::DenseElementsAttr::getFromRawBuffer (
                    tensor_type,
                    llvm::ArrayRef<char> (
                        reinterpret_cast<const char*> (v->data().data()),
                        v->data().size()));

            mlir::Value const_val =
                mlir::arith::ConstantOp::create (b, loc, tensor_type, dense_attr);

            value_map_[v->name()] = const_val;
        }
    }

    // -----------------------------------------------------------------------
    // node dispatch
    // -----------------------------------------------------------------------

    void emit_node (mlir::OpBuilder& b, mlir::Location loc, const Node& node)
    {
        const std::string& op = node.op_type();

        if (op == "Add")  { emit_elementwise (b, loc, node, op); return; }
        if (op == "Mul")  { emit_elementwise (b, loc, node, op); return; }
        if (op == "Relu") { emit_relu        (b, loc, node);     return; }

        throw std::runtime_error ("emit_node: unsupported op '" + op + "'");
    }

    // -----------------------------------------------------------------------
    // op emitters
    // -----------------------------------------------------------------------

    mlir::Value lookup (const std::string& name)
    {
        auto it = value_map_.find (name);
        if (it == value_map_.end())
            throw std::runtime_error ("lookup: value not found: " + name);

        return it->second;
    }

    void emit_elementwise (mlir::OpBuilder& b, mlir::Location loc,
                           const Node& node, const std::string& op_type)
    {
        (void) b; (void) loc; (void) node; (void) op_type;
        throw std::runtime_error ("emit_elementwise: not yet implemented");
    }

    void emit_relu (mlir::OpBuilder& b, mlir::Location loc, const Node& node)
    {
        (void) b; (void) loc; (void) node;
        throw std::runtime_error ("emit_relu: not yet implemented");
    }
};

} // namespace

// ---------------------------------------------------------------------------
// public entry point
// ---------------------------------------------------------------------------

mlir::OwningOpRef<mlir::ModuleOp> graph_to_mlir (const GraphBuilder& builder)
{
    MLIRCodegen codegen (builder);

    return codegen.emit();
}

} // namespace tc
