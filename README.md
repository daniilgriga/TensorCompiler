![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![macOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0)
![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-%23008FBA.svg?style=for-the-badge&logo=cmake&logoColor=white)

# Tensor Compiler

The goal of the project is a full-fledged tensor compiler.
Currently the **frontend** stage is implemented:
- import ONNX model into an internal compute graph (`GraphBuilder`)
- validate graph consistency (`verify`)
- export graph to Graphviz `.dot` for visualization

## How to Install

```bash
git clone git@github.com:daniilgriga/TensorCompiler.git
cd TensorCompiler/
```

## How to Build

```bash
# Debug + tests
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
cmake --build build/debug

# Release
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
cmake --build build/release
```

## How to Run

### Print Graph Stats

```bash
./build/debug/tc_main tests/models/conv_relu_gemm.onnx
```

### Export Graph to DOT

```bash
./build/debug/tc_main tests/models/conv_relu_gemm.onnx --dot output/conv_relu_gemm.dot
```

## Graph Visualization (SVG)

Convert generated DOT to SVG with GraphViz:

```bash
dot -Tsvg output/conv_relu_gemm.dot -o output/conv_relu_gemm.svg
```

Project already contains sample visualizations:
- `output/conv_relu_gemm.svg` — small gen test model (Conv -> Relu -> Reshape -> Gemm):

<p align="center">
  <img src="output/conv_relu_gemm.svg" alt="conv_relu_gemm graph">
</p>

- `output/adv_inception_v3.svg` — [Adversarial Inception v3](https://github.com/onnx/models) from the ONNX Model Zoo: [watch it here](https://github.com/daniilgriga/TensorCompiler/blob/dev/output/adv_inception_v3.svg)

## Tests

### Build Tests

```bash
cmake -S . -B build/tests -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
cmake --build build/tests
```

### Run Tests

```bash
ctest --test-dir build/tests --output-on-failure
```
