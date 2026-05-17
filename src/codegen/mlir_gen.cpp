#include <cstring>
#include <limits>
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
        : builder_ (builder),
          ctx_ (std::make_unique<mlir::MLIRContext>())
    {
        // register all dialects we emit
        ctx_->loadDialect<mlir::func::FuncDialect,
                          mlir::arith::ArithDialect,
                          mlir::linalg::LinalgDialect,
                          mlir::tensor::TensorDialect>();
    }

    MlirModule emit ()
    {
        auto loc = mlir::UnknownLoc::get (ctx_.get());

        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::ModuleOp::create (loc);

        mlir::OpBuilder b (module->getBodyRegion());

        emit_func (b, loc, *module);

        return MlirModule {std::move (ctx_), std::move (module)};
    }

private:
    const GraphBuilder& builder_;
    std::unique_ptr<mlir::MLIRContext> ctx_;

    // SSA values for graph values by name
    std::unordered_map<std::string, mlir::Value> value_map_;

    // -----------------------------------------------------------------------
    // func.func emission
    // -----------------------------------------------------------------------

    void emit_func (mlir::OpBuilder& b,
                    mlir::Location   loc,
                    mlir::ModuleOp   module)
    {
        // input types are always known
        llvm::SmallVector<mlir::Type> arg_types;
        for (const Value* v : builder_.graph_inputs())
            arg_types.push_back (value_to_tensor_type (*v, ctx_.get()));

        auto placeholder_func_type = b.getFunctionType (arg_types, {});
        auto func_op = mlir::func::FuncOp::create (loc, "main", placeholder_func_type);

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

        // collect outputs - types are now known from value_map_ SSA values
        llvm::SmallVector<mlir::Value> results;
        llvm::SmallVector<mlir::Type>  res_types;
        for (const Value* v : builder_.graph_outputs())
        {
            auto it = value_map_.find (v->name());
            if (it == value_map_.end())
                throw std::runtime_error (
                    "emit_func: output value not found: " + v->name());
            results.push_back (it->second);
            res_types.push_back (it->second.getType());
        }

        mlir::func::ReturnOp::create (body_builder, loc, results);

        // patch the function type with the correct return types
        func_op.setType (b.getFunctionType (arg_types, res_types));

        module.push_back (func_op);
    }

    void emit_initializers (mlir::OpBuilder& b, mlir::Location loc)
    {
        for (const Value* v : builder_.graph_initializers())
        {
            if (value_map_.count (v->name()))
                continue; // already emitted

            mlir::RankedTensorType tensor_type = value_to_tensor_type (*v, ctx_.get());

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

        if (op == "Add")     { emit_elementwise (b, loc, node, op); return; }
        if (op == "Mul")     { emit_elementwise (b, loc, node, op); return; }
        if (op == "Relu")    { emit_relu        (b, loc, node);     return; }
        if (op == "MatMul")  { emit_matmul      (b, loc, node);     return; }
        if (op == "Gemm")    { emit_gemm        (b, loc, node);     return; }
        if (op == "Conv")    { emit_conv        (b, loc, node);     return; }
        if (op == "Reshape") { emit_reshape     (b, loc, node);     return; }
        if (op == "MaxPool") { emit_maxpool     (b, loc, node);     return; }

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

    // output slot tensor required by linalg value semantics
    mlir::Value make_empty_like (mlir::OpBuilder& b, mlir::Location loc,
                                 mlir::Value ref)
    {
        auto tensor_type = mlir::cast<mlir::RankedTensorType> (ref.getType());

        // dynamic dims need runtime tensor.dim values; static ones are in the type
        llvm::SmallVector<mlir::Value> dyn_sizes;
        for (int64_t i = 0; i < tensor_type.getRank(); ++i)
        {
            if (tensor_type.isDynamicDim (i))
            {
                mlir::Value idx = mlir::arith::ConstantIndexOp::create (
                    b, loc, i);
                dyn_sizes.push_back (
                    mlir::tensor::DimOp::create (b, loc, ref, idx));
            }
        }

        return mlir::tensor::EmptyOp::create (
            b, loc, tensor_type.getShape(), tensor_type.getElementType(),
            dyn_sizes);
    }

    void emit_elementwise (mlir::OpBuilder& b, mlir::Location loc,
                           const Node& node, const std::string& op_type)
    {
        if (node.inputs().size() < 2 || node.outputs().size() < 1)
            throw std::runtime_error (
                "emit_elementwise: expected 2 inputs and 1 output for op '" +
                op_type + "'");

        mlir::Value lhs = lookup (node.inputs()[0]->name());
        mlir::Value rhs = lookup (node.inputs()[1]->name());
        mlir::Value out = make_empty_like (b, loc, lhs);

        mlir::Value result;
        if (op_type == "Add")
            result = mlir::linalg::AddOp::create (b, loc, mlir::ValueRange{lhs, rhs},
                                                  mlir::ValueRange{out}).getResult (0);
        else if (op_type == "Mul")
            result = mlir::linalg::MulOp::create (b, loc, mlir::ValueRange{lhs, rhs},
                                                  mlir::ValueRange{out}).getResult (0);
        else
            throw std::runtime_error (
                "emit_elementwise: unknown op_type '" + op_type + "'");

        value_map_[node.outputs()[0]->name()] = result;
    }

    // zero-filled tensor of given shape and element type - required as output
    // slot for linalg.matmul because it accumulates: out[i,j] += a[i,k]*b[k,j]
    mlir::Value make_zero_tensor (mlir::OpBuilder& b, mlir::Location loc,
                                  llvm::ArrayRef<int64_t> shape, mlir::Type elem)
    {
        auto type = mlir::RankedTensorType::get (shape, elem);
        mlir::Attribute zero_attr;
        if (llvm::isa<mlir::FloatType> (elem))
            zero_attr = b.getFloatAttr (elem, 0.0);
        else
            zero_attr = b.getIntegerAttr (elem, 0);

        auto splat = mlir::SplatElementsAttr::get (type, zero_attr);

        return mlir::arith::ConstantOp::create (b, loc, type, splat);
    }

    // tensor filled with -inf (float) or INT_MIN (int) - init slot for max-pooling
    mlir::Value make_neg_inf_tensor (mlir::OpBuilder& b, mlir::Location loc,
                                     llvm::ArrayRef<int64_t> shape, mlir::Type elem)
    {
        auto type = mlir::RankedTensorType::get (shape, elem);
        mlir::Attribute val_attr;
        if (llvm::isa<mlir::FloatType> (elem))
            val_attr = b.getFloatAttr (elem, -std::numeric_limits<double>::infinity());
        else
            val_attr = b.getIntegerAttr (
                elem,
                llvm::APInt::getSignedMinValue (
                    llvm::cast<mlir::IntegerType> (elem).getWidth()).getSExtValue());

        auto splat = mlir::SplatElementsAttr::get (type, val_attr);

        return mlir::arith::ConstantOp::create (b, loc, type, splat);
    }

    void emit_matmul (mlir::OpBuilder& b, mlir::Location loc, const Node& node)
    {
        if (node.inputs().size() < 2 || node.outputs().size() < 1)
            throw std::runtime_error ("emit_matmul: expected 2 inputs and 1 output");

        mlir::Value a = lookup (node.inputs()[0]->name());
        mlir::Value bv = lookup (node.inputs()[1]->name());

        auto a_type  = mlir::cast<mlir::RankedTensorType> (a.getType());
        auto bv_type = mlir::cast<mlir::RankedTensorType> (bv.getType());

        // output shape [M, N]: M from a's first dim, N from b's last dim
        int64_t M = a_type.getShape()[0];
        int64_t N = bv_type.getShape()[1];
        mlir::Type elem = a_type.getElementType();

        mlir::Value out = make_zero_tensor (b, loc, {M, N}, elem);

        mlir::Value result =
            mlir::linalg::MatmulOp::create (
                b, loc,
                mlir::ValueRange{a, bv},
                mlir::ValueRange{out})
            .getResult (0);

        value_map_[node.outputs()[0]->name()] = result;
    }

    // scale all elements of a tensor by a float scalar via linalg.generic
    mlir::Value scale_tensor (mlir::OpBuilder& b, mlir::Location loc,
                              mlir::Value tensor, double factor)
    {
        auto t    = mlir::cast<mlir::RankedTensorType> (tensor.getType());
        mlir::Type elem  = t.getElementType();
        int64_t    rank  = t.getRank();

        mlir::AffineMap identity =
            mlir::AffineMap::getMultiDimIdentityMap (rank, ctx_.get());
        llvm::SmallVector<mlir::utils::IteratorType> iters (
            rank, mlir::utils::IteratorType::parallel);

        mlir::Value out = make_empty_like (b, loc, tensor);

        auto generic = mlir::linalg::GenericOp::create (
            b, loc,
            mlir::TypeRange{t},
            mlir::ValueRange{tensor},
            mlir::ValueRange{out},
            llvm::SmallVector<mlir::AffineMap>{identity, identity},
            iters, "", "");

        mlir::Block* body = &generic.getRegion().emplaceBlock();
        body->addArgument (elem, loc);
        body->addArgument (elem, loc);

        mlir::OpBuilder bb (body, body->end());
        mlir::Value scalar = mlir::arith::ConstantOp::create (
                                 bb, loc, elem,
                                 bb.getFloatAttr (elem, factor)).getResult();
        mlir::Value result = mlir::arith::MulFOp::create (
                                 bb, loc, body->getArgument (0), scalar).getResult();
        mlir::linalg::YieldOp::create (bb, loc, mlir::ValueRange{result});

        return generic.getResult (0);
    }

    void emit_gemm (mlir::OpBuilder& b, mlir::Location loc, const Node& node)
    {
        if (node.inputs().size() < 2 || node.outputs().size() < 1)
            throw std::runtime_error ("emit_gemm: expected at least 2 inputs and 1 output");

        mlir::Value a  = lookup (node.inputs()[0]->name());
        mlir::Value bv = lookup (node.inputs()[1]->name());

        auto a_type  = mlir::cast<mlir::RankedTensorType> (a.getType());
        auto bv_type = mlir::cast<mlir::RankedTensorType> (bv.getType());

        float   alpha  = node.attr_as<float>   ("alpha") .value_or (1.0f);
        float   beta   = node.attr_as<float>   ("beta")  .value_or (1.0f);
        int64_t transA = node.attr_as<int64_t> ("transA").value_or (0);
        int64_t transB = node.attr_as<int64_t> ("transB").value_or (0);

        // output shape [M, N]
        int64_t M = transA ? a_type.getShape()[1] : a_type.getShape()[0];
        int64_t N = transB ? bv_type.getShape()[0] : bv_type.getShape()[1];
        mlir::Type elem = a_type.getElementType();

        mlir::Value out = make_zero_tensor (b, loc, {M, N}, elem);

        mlir::Value result;
        if (!transA && !transB)
            result = mlir::linalg::MatmulOp::create (
                         b, loc, mlir::ValueRange{a, bv},
                         mlir::ValueRange{out}).getResult (0);
        else if (!transA && transB)
            result = mlir::linalg::MatmulTransposeBOp::create (
                         b, loc, mlir::ValueRange{a, bv},
                         mlir::ValueRange{out}).getResult (0);
        else if (transA && !transB)
            result = mlir::linalg::MatmulTransposeAOp::create (
                         b, loc, mlir::ValueRange{a, bv},
                         mlir::ValueRange{out}).getResult (0);
        else
            throw std::runtime_error ("emit_gemm: transA=1 and transB=1 not supported");

        if (alpha != 1.0f)
            result = scale_tensor (b, loc, result, static_cast<double> (alpha));

        // add bias C if present
        if (node.inputs().size() >= 3 && !node.inputs()[2]->name().empty())
        {
            mlir::Value bias = lookup (node.inputs()[2]->name());

            if (beta != 1.0f && beta != 0.0f)
                bias = scale_tensor (b, loc, bias, static_cast<double> (beta));

            if (beta != 0.0f)
            {
                // bias is rank-1 [N], result is rank-2 [M, N] - need broadcast via generic
                auto res_type = mlir::cast<mlir::RankedTensorType> (result.getType());
                mlir::Value bias_out = make_empty_like (b, loc, result);

                mlir::AffineMap res_map =
                    mlir::AffineMap::getMultiDimIdentityMap (2, ctx_.get());
                // bias map: (d0, d1) -> (d1) - broadcast over d0
                mlir::AffineMap bias_map =
                    mlir::AffineMap::get (2, 0,
                        {mlir::getAffineDimExpr (1, ctx_.get())}, ctx_.get());

                llvm::SmallVector<mlir::utils::IteratorType> iters (
                    2, mlir::utils::IteratorType::parallel);

                auto generic = mlir::linalg::GenericOp::create (
                    b, loc,
                    mlir::TypeRange{res_type},
                    mlir::ValueRange{result, bias},
                    mlir::ValueRange{bias_out},
                    llvm::SmallVector<mlir::AffineMap>{res_map, bias_map, res_map},
                    iters, "", "");

                mlir::Type elem = res_type.getElementType();
                mlir::Block* body = &generic.getRegion().emplaceBlock();
                body->addArgument (elem, loc);
                body->addArgument (elem, loc);
                body->addArgument (elem, loc);

                mlir::OpBuilder bb (body, body->end());
                mlir::Value sum = mlir::arith::AddFOp::create (
                                      bb, loc,
                                      body->getArgument (0),
                                      body->getArgument (1)).getResult();
                mlir::linalg::YieldOp::create (bb, loc, mlir::ValueRange{sum});

                result = generic.getResult (0);
            }
        }

        value_map_[node.outputs()[0]->name()] = result;
    }

    void emit_conv (mlir::OpBuilder& b, mlir::Location loc, const Node& node)
    {
        if (node.inputs().size() < 2 || node.outputs().size() < 1)
            throw std::runtime_error ("emit_conv: expected at least 2 inputs and 1 output");

        mlir::Value input  = lookup (node.inputs()[0]->name());
        mlir::Value filter = lookup (node.inputs()[1]->name());

        auto input_type  = mlir::cast<mlir::RankedTensorType> (input.getType());
        auto filter_type = mlir::cast<mlir::RankedTensorType> (filter.getType());

        llvm::ArrayRef<int64_t> in_shape = input_type.getShape();  // [N, C, H, W]
        llvm::ArrayRef<int64_t> f_shape  = filter_type.getShape(); // [F, C, Kh, Kw]

        // read attributes with ONNX defaults
        using Ints = std::vector<int64_t>;
        auto strides   = node.attr_as<Ints> ("strides")  .value_or (Ints{1, 1});
        auto dilations = node.attr_as<Ints> ("dilations").value_or (Ints{1, 1});
        auto pads      = node.attr_as<Ints> ("pads")     .value_or (Ints{0, 0, 0, 0});
        // pads: [top, left, bottom, right]

        int64_t N  = in_shape[0];
        int64_t H  = in_shape[2], W  = in_shape[3];
        int64_t F  = f_shape[0];
        int64_t Kh = f_shape[2], Kw = f_shape[3];

        int64_t Ho = (H + pads[0] + pads[2] - dilations[0] * (Kh - 1) - 1) / strides[0] + 1;
        int64_t Wo = (W + pads[1] + pads[3] - dilations[1] * (Kw - 1) - 1) / strides[1] + 1;

        mlir::Type elem = input_type.getElementType();

        // apply padding if needed - PadOp takes low/high as ValueRange of index SSA values
        if (pads[0] || pads[1] || pads[2] || pads[3])
        {
            llvm::SmallVector<int64_t> padded_shape = {
                N, in_shape[1],
                H + pads[0] + pads[2],
                W + pads[1] + pads[3]
            };
            auto padded_type = mlir::RankedTensorType::get (padded_shape, elem);

            llvm::SmallVector<mlir::Value> low_vals  = {
                mlir::arith::ConstantIndexOp::create (b, loc, 0),
                mlir::arith::ConstantIndexOp::create (b, loc, 0),
                mlir::arith::ConstantIndexOp::create (b, loc, pads[0]),
                mlir::arith::ConstantIndexOp::create (b, loc, pads[1])
            };

            llvm::SmallVector<mlir::Value> high_vals = {
                mlir::arith::ConstantIndexOp::create (b, loc, 0),
                mlir::arith::ConstantIndexOp::create (b, loc, 0),
                mlir::arith::ConstantIndexOp::create (b, loc, pads[2]),
                mlir::arith::ConstantIndexOp::create (b, loc, pads[3])
            };

            auto pad_op = mlir::tensor::PadOp::create (
                b, loc, padded_type, input,
                mlir::ValueRange (low_vals), mlir::ValueRange (high_vals));

            mlir::Block* body = &pad_op.getRegion().emplaceBlock();
            for (int64_t i = 0; i < 4; ++i)
                body->addArgument (b.getIndexType(), loc);

            mlir::OpBuilder bb (body, body->end());
            mlir::Value zero = mlir::arith::ConstantOp::create (
                                   bb, loc, elem,
                                   bb.getFloatAttr (elem, 0.0)).getResult();
            mlir::tensor::YieldOp::create (bb, loc, zero);

            input = pad_op.getResult();
        }

        mlir::Value out = make_zero_tensor (b, loc, {N, F, Ho, Wo}, elem);

        // overload: (builder, loc, TypeRange resultTypes, inputs, outputs, strides, dilations)
        mlir::Value result =
            mlir::linalg::Conv2DNchwFchwOp::create (
                b, loc,
                mlir::TypeRange{mlir::RankedTensorType::get ({N, F, Ho, Wo}, elem)},
                mlir::ValueRange{input, filter},
                mlir::ValueRange{out},
                b.getDenseI64ArrayAttr (strides),
                b.getDenseI64ArrayAttr (dilations)).getResult (0);

        // add bias if present (broadcast [F] over [N, F, Ho, Wo])
        if (node.inputs().size() >= 3 && !node.inputs()[2]->name().empty())
        {
            mlir::Value bias     = lookup (node.inputs()[2]->name());
            mlir::Value bias_out = make_empty_like (b, loc, result);

            // broadcast via linalg.generic: bias[f] added to result[n,f,h,w]
            auto res_type  = mlir::cast<mlir::RankedTensorType> (result.getType());
            mlir::AffineMap res_map  =
                mlir::AffineMap::getMultiDimIdentityMap (4, ctx_.get());
            // bias indexing map: (n, f, h, w) -> (f)  - dimension 1 only
            mlir::AffineMap bias_map =
                mlir::AffineMap::get (4, 0,
                    {mlir::getAffineDimExpr (1, ctx_.get())}, ctx_.get());

            llvm::SmallVector<mlir::utils::IteratorType> iters (
                4, mlir::utils::IteratorType::parallel);

            auto generic = mlir::linalg::GenericOp::create (
                b, loc,
                mlir::TypeRange{res_type},
                mlir::ValueRange{result, bias},
                mlir::ValueRange{bias_out},
                llvm::SmallVector<mlir::AffineMap>{res_map, bias_map, res_map},
                iters, "", "");

            mlir::Block* body = &generic.getRegion().emplaceBlock();
            body->addArgument (elem, loc); // result scalar
            body->addArgument (elem, loc); // bias scalar
            body->addArgument (elem, loc); // output scalar

            mlir::OpBuilder bb (body, body->end());
            mlir::Value sum = mlir::arith::AddFOp::create (
                                  bb, loc,
                                  body->getArgument (0),
                                  body->getArgument (1)).getResult();
            mlir::linalg::YieldOp::create (bb, loc, mlir::ValueRange{sum});

            result = generic.getResult (0);
        }

        value_map_[node.outputs()[0]->name()] = result;
    }

    void emit_reshape (mlir::OpBuilder& b, mlir::Location loc, const Node& node)
    {
        if (node.inputs().size() < 2 || node.outputs().size() < 1)
            throw std::runtime_error ("emit_reshape: expected 2 inputs and 1 output");

        mlir::Value input = lookup (node.inputs()[0]->name());
        const Value* shape_val = node.inputs()[1];

        // shape input must be a static int64 initializer
        const auto& shape_bytes = shape_val->data();
        if (shape_bytes.empty())
            throw std::runtime_error (
                "emit_reshape: shape input '" + shape_val->name() +
                "' has no data - dynamic shapes not supported");

        if (shape_bytes.size() % sizeof (int64_t) != 0)
            throw std::runtime_error (
                "emit_reshape: shape input '" + shape_val->name() +
                "' has " + std::to_string (shape_bytes.size()) +
                " bytes, not a multiple of sizeof(int64_t)");

        const int64_t shape_rank =
            static_cast<int64_t> (shape_bytes.size() / sizeof (int64_t));

        llvm::SmallVector<int64_t> target_shape (
            static_cast<std::size_t> (shape_rank));
        std::memcpy (target_shape.data(), shape_bytes.data(), shape_bytes.size());

        // build index-typed shape tensor: tensor.reshape requires <Nxindex>
        mlir::Type idx_type = b.getIndexType();
        auto shape_tensor_type =
            mlir::RankedTensorType::get ({shape_rank}, idx_type);

        llvm::SmallVector<mlir::Attribute> idx_attrs;
        idx_attrs.reserve (static_cast<std::size_t> (shape_rank));
        for (int64_t v : target_shape)
            idx_attrs.push_back (b.getIndexAttr (v));

        mlir::Value shape_tensor = mlir::arith::ConstantOp::create (
            b, loc, shape_tensor_type,
            mlir::DenseElementsAttr::get (shape_tensor_type,
                                          llvm::ArrayRef<mlir::Attribute> (idx_attrs)));

        auto input_type = mlir::cast<mlir::RankedTensorType> (input.getType());
        auto result_type =
            mlir::RankedTensorType::get (target_shape, input_type.getElementType());

        mlir::Value result =
            mlir::tensor::ReshapeOp::create (b, loc, result_type, input,
                shape_tensor).getResult();

        value_map_[node.outputs()[0]->name()] = result;
    }

    // Relu = max(x, 0) via linalg.generic - no dedicated linalg.relu exists
    void emit_relu (mlir::OpBuilder& b, mlir::Location loc, const Node& node)
    {
        if (node.inputs().size() < 1 || node.outputs().size() < 1)
            throw std::runtime_error ("emit_relu: expected 1 input and 1 output");

        mlir::Value input = lookup (node.inputs()[0]->name());
        mlir::Value out   = make_empty_like (b, loc, input);

        auto tensor_type =
            mlir::cast<mlir::RankedTensorType> (input.getType());
        mlir::Type elem = tensor_type.getElementType();

        int64_t rank = tensor_type.getRank();

        mlir::AffineMap identity =
            mlir::AffineMap::getMultiDimIdentityMap (rank, ctx_.get());
        llvm::SmallVector<mlir::AffineMap> indexing_maps = {identity, identity};

        llvm::SmallVector<mlir::utils::IteratorType> iterator_types (
            rank, mlir::utils::IteratorType::parallel);

        auto generic_op = mlir::linalg::GenericOp::create (
            b, loc,
            /*resultTensorTypes=*/mlir::TypeRange{tensor_type},
            /*inputs=*/mlir::ValueRange{input},
            /*outputs=*/mlir::ValueRange{out},
            indexing_maps,
            iterator_types,
            /*doc=*/"",
            /*library_call=*/"");

        mlir::Block* body = &generic_op.getRegion().emplaceBlock();
        body->addArgument (elem, loc); // input scalar
        body->addArgument (elem, loc); // output scalar (required by linalg.generic)

        mlir::OpBuilder body_b (body, body->end());

        mlir::Value in_scalar  = body->getArgument (0);
        mlir::Value zero;
        mlir::Value max_val;

        if (elem.isF32() || elem.isF64() || elem.isF16() || elem.isBF16())
        {
            zero    = mlir::arith::ConstantOp::create (
                          body_b, loc, elem,
                          body_b.getFloatAttr (elem, 0.0)).getResult();
            max_val = mlir::arith::MaximumFOp::create (
                          body_b, loc, in_scalar, zero).getResult();
        }
        else
        {
            zero    = mlir::arith::ConstantOp::create (
                          body_b, loc, elem,
                          body_b.getIntegerAttr (elem, 0)).getResult();
            max_val = mlir::arith::MaxSIOp::create (
                          body_b, loc, in_scalar, zero).getResult();
        }

        mlir::linalg::YieldOp::create (body_b, loc, mlir::ValueRange{max_val});

        value_map_[node.outputs()[0]->name()] = generic_op.getResult (0);
    }

    void emit_maxpool (mlir::OpBuilder& b, mlir::Location loc, const Node& node)
    {
        if (node.inputs().size() < 1 || node.outputs().size() < 1)
            throw std::runtime_error ("emit_maxpool: expected 1 input and 1 output");

        mlir::Value input = lookup (node.inputs()[0]->name());
        auto in_type = mlir::cast<mlir::RankedTensorType> (input.getType());
        auto in_shape = in_type.getShape();   // [N, C, H, W]
        mlir::Type elem = in_type.getElementType();

        using Ints = std::vector<int64_t>;
        auto kernel_shape = node.attr_as<Ints> ("kernel_shape").value_or (Ints{1, 1});
        auto strides      = node.attr_as<Ints> ("strides")     .value_or (Ints{1, 1});
        auto dilations    = node.attr_as<Ints> ("dilations")   .value_or (Ints{1, 1});
        auto pads         = node.attr_as<Ints> ("pads")        .value_or (Ints{0, 0, 0, 0});
        // pads: [top, left, bottom, right]

        int64_t N = in_shape[0], C  = in_shape[1];
        int64_t H = in_shape[2], W  = in_shape[3];
        int64_t Kh = kernel_shape[0], Kw = kernel_shape[1];

        int64_t Ho = (H + pads[0] + pads[2] - dilations[0] * (Kh - 1) - 1) / strides[0] + 1;
        int64_t Wo = (W + pads[1] + pads[3] - dilations[1] * (Kw - 1) - 1) / strides[1] + 1;

        // pad with -inf so padded positions don't affect the max reduction
        if (pads[0] || pads[1] || pads[2] || pads[3])
        {
            llvm::SmallVector<int64_t> padded_shape = {
                N, C,
                H + pads[0] + pads[2],
                W + pads[1] + pads[3]
            };
            auto padded_type = mlir::RankedTensorType::get (padded_shape, elem);

            llvm::SmallVector<mlir::Value> low_vals = {
                mlir::arith::ConstantIndexOp::create (b, loc, 0),
                mlir::arith::ConstantIndexOp::create (b, loc, 0),
                mlir::arith::ConstantIndexOp::create (b, loc, pads[0]),
                mlir::arith::ConstantIndexOp::create (b, loc, pads[1])
            };
            llvm::SmallVector<mlir::Value> high_vals = {
                mlir::arith::ConstantIndexOp::create (b, loc, 0),
                mlir::arith::ConstantIndexOp::create (b, loc, 0),
                mlir::arith::ConstantIndexOp::create (b, loc, pads[2]),
                mlir::arith::ConstantIndexOp::create (b, loc, pads[3])
            };

            auto pad_op = mlir::tensor::PadOp::create (
                b, loc, padded_type, input,
                mlir::ValueRange (low_vals), mlir::ValueRange (high_vals));

            mlir::Block* body = &pad_op.getRegion().emplaceBlock();
            for (int64_t i = 0; i < 4; ++i)
                body->addArgument (b.getIndexType(), loc);

            mlir::OpBuilder bb (body, body->end());
            mlir::Value neg_inf = mlir::arith::ConstantOp::create (
                bb, loc, elem,
                bb.getFloatAttr (elem, -std::numeric_limits<double>::infinity())).getResult();
            mlir::tensor::YieldOp::create (bb, loc, neg_inf);

            input = pad_op.getResult();
        }

        // fake kernel [Kh, Kw] - linalg.pooling_nchw_max needs it only for shape
        mlir::Value fake_kernel = make_zero_tensor (b, loc, {Kh, Kw}, elem);

        // output initialized with -inf so max accumulation is correct
        mlir::Value out     = make_neg_inf_tensor (b, loc, {N, C, Ho, Wo}, elem);
        auto        out_type = mlir::RankedTensorType::get ({N, C, Ho, Wo}, elem);

        mlir::Value result =
            mlir::linalg::PoolingNchwMaxOp::create (
                b, loc,
                mlir::TypeRange{out_type},
                mlir::ValueRange{input, fake_kernel},
                mlir::ValueRange{out},
                b.getDenseI64ArrayAttr (strides),
                b.getDenseI64ArrayAttr (dilations))
            .getResult (0);

        value_map_[node.outputs()[0]->name()] = result;
    }
};

} // namespace

// ---------------------------------------------------------------------------
// public entry point
// ---------------------------------------------------------------------------

MlirModule graph_to_mlir (const GraphBuilder& builder)
{
    MLIRCodegen codegen (builder);

    return codegen.emit();
}

} // namespace tc
