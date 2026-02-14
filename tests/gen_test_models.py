import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def make_conv_relu_gemm():
    """
    Conv -> Relu -> Reshape -> Gemm

    4 nodes - tests:
      - Conv attributes: kernel_shape, strides, pads
      - Relu: no attributes
      - Gemm attributes: transB, alpha, beta
      - Reshape: non-mandatory op (tests unknown handling)
    """

    # inputs
    X = helper.make_tensor_value_info ("X", TensorProto.FLOAT, [1, 1, 5, 5])

    # weights (initializers)
    W_conv = numpy_helper.from_array (
        np.ones ((1, 1, 3, 3), dtype=np.float32), name="W_conv")

    B_conv = numpy_helper.from_array (
        np.zeros (1, dtype=np.float32), name="B_conv")

    W_gemm = numpy_helper.from_array (
        np.ones ((9, 10), dtype=np.float32), name="W_gemm")

    B_gemm = numpy_helper.from_array (
        np.zeros (10, dtype=np.float32), name="B_gemm")

    # Shape for Reshape node
    shape_const = numpy_helper.from_array (
        np.array ([1, 9], dtype=np.int64), name="flat_shape")

    # nodes
    conv = helper.make_node (
        "Conv", ["X", "W_conv", "B_conv"], ["conv1_out"],
        name="conv1",
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[0, 0, 0, 0])

    relu = helper.make_node (
        "Relu", ["conv1_out"], ["relu1_out"],
        name="relu1")

    reshape = helper.make_node (
        "Reshape", ["relu1_out", "flat_shape"], ["flat"],
        name="reshape1")

    gemm = helper.make_node (
        "Gemm", ["flat", "W_gemm", "B_gemm"], ["Y"],
        name="gemm1",
        transB=1,
        alpha=1.0,
        beta=1.0)

    # output
    Y = helper.make_tensor_value_info ("Y", TensorProto.FLOAT, [1, 10])

    # graph
    graph = helper.make_graph (
        [conv, relu, reshape, gemm],
        "test_conv_relu_gemm",
        inputs=[X],
        outputs=[Y],
        initializer=[W_conv, B_conv, W_gemm, B_gemm, shape_const])

    model = helper.make_model (graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    onnx.checker.check_model (model)
    onnx.save (model, "tests/models/conv_relu_gemm.onnx")
    print ("saved tests/models/conv_relu_gemm.onnx")

if __name__ == "__main__":
    make_conv_relu_gemm()
