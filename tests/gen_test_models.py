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

def make_add():
    A = helper.make_tensor_value_info ("A", TensorProto.FLOAT, [2, 3])
    B = helper.make_tensor_value_info ("B", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info ("Y", TensorProto.FLOAT, [2, 3])
    node = helper.make_node ("Add", ["A", "B"], ["Y"])
    graph = helper.make_graph ([node], "add", [A, B], [Y])
    model = helper.make_model (graph, opset_imports=[helper.make_opsetid ("", 17)])
    model.ir_version = 8
    onnx.checker.check_model (model)
    onnx.save (model, "tests/models/add.onnx")
    print ("saved tests/models/add.onnx")


def make_mul():
    A = helper.make_tensor_value_info ("A", TensorProto.FLOAT, [2, 3])
    B = helper.make_tensor_value_info ("B", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info ("Y", TensorProto.FLOAT, [2, 3])
    node = helper.make_node ("Mul", ["A", "B"], ["Y"])
    graph = helper.make_graph ([node], "mul", [A, B], [Y])
    model = helper.make_model (graph, opset_imports=[helper.make_opsetid ("", 17)])
    model.ir_version = 8
    onnx.checker.check_model (model)
    onnx.save (model, "tests/models/mul.onnx")
    print ("saved tests/models/mul.onnx")


def make_relu():
    X = helper.make_tensor_value_info ("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info ("Y", TensorProto.FLOAT, [2, 3])
    node = helper.make_node ("Relu", ["X"], ["Y"])
    graph = helper.make_graph ([node], "relu", [X], [Y])
    model = helper.make_model (graph, opset_imports=[helper.make_opsetid ("", 17)])
    model.ir_version = 8
    onnx.checker.check_model (model)
    onnx.save (model, "tests/models/relu.onnx")
    print ("saved tests/models/relu.onnx")


def make_matmul():
    A = helper.make_tensor_value_info ("A", TensorProto.FLOAT, [4, 8])
    B = helper.make_tensor_value_info ("B", TensorProto.FLOAT, [8, 16])
    Y = helper.make_tensor_value_info ("Y", TensorProto.FLOAT, [4, 16])
    node = helper.make_node ("MatMul", ["A", "B"], ["Y"])
    graph = helper.make_graph ([node], "matmul", [A, B], [Y])
    model = helper.make_model (graph, opset_imports=[helper.make_opsetid ("", 17)])
    model.ir_version = 8
    onnx.checker.check_model (model)
    onnx.save (model, "tests/models/matmul.onnx")
    print ("saved tests/models/matmul.onnx")


def make_gemm():
    A  = helper.make_tensor_value_info ("A", TensorProto.FLOAT, [4, 8])
    Y  = helper.make_tensor_value_info ("Y", TensorProto.FLOAT, [4, 16])
    W  = numpy_helper.from_array (np.zeros ((16, 8),  dtype=np.float32), name="W")
    Bv = numpy_helper.from_array (np.zeros ((16,),    dtype=np.float32), name="B")
    node = helper.make_node ("Gemm", ["A", "W", "B"], ["Y"],
                             transB=1, alpha=1.0, beta=1.0)
    graph = helper.make_graph ([node], "gemm", [A], [Y], initializer=[W, Bv])
    model = helper.make_model (graph, opset_imports=[helper.make_opsetid ("", 17)])
    model.ir_version = 8
    onnx.checker.check_model (model)
    onnx.save (model, "tests/models/gemm.onnx")
    print ("saved tests/models/gemm.onnx")


def make_conv():
    X = helper.make_tensor_value_info ("X", TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info ("Y", TensorProto.FLOAT, [1, 1, 3, 3])
    W = numpy_helper.from_array (np.zeros ((1, 1, 3, 3), dtype=np.float32), name="W")
    node = helper.make_node ("Conv", ["X", "W"], ["Y"],
                             kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0])
    graph = helper.make_graph ([node], "conv", [X], [Y], initializer=[W])
    model = helper.make_model (graph, opset_imports=[helper.make_opsetid ("", 17)])
    model.ir_version = 8
    onnx.checker.check_model (model)
    onnx.save (model, "tests/models/conv.onnx")
    print ("saved tests/models/conv.onnx")


def make_conv_pad():
    X = helper.make_tensor_value_info ("X", TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info ("Y", TensorProto.FLOAT, [1, 1, 5, 5])
    W = numpy_helper.from_array (np.zeros ((1, 1, 3, 3), dtype=np.float32), name="W")
    node = helper.make_node ("Conv", ["X", "W"], ["Y"],
                             kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])
    graph = helper.make_graph ([node], "conv_pad", [X], [Y], initializer=[W])
    model = helper.make_model (graph, opset_imports=[helper.make_opsetid ("", 17)])
    model.ir_version = 8
    onnx.checker.check_model (model)
    onnx.save (model, "tests/models/conv_pad.onnx")
    print ("saved tests/models/conv_pad.onnx")


if __name__ == "__main__":
    make_conv_relu_gemm()
    make_add()
    make_mul()
    make_relu()
    make_matmul()
    make_gemm()
    make_conv()
    make_conv_pad()
