![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![macOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0)
![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-%23008FBA.svg?style=for-the-badge&logo=cmake&logoColor=white)

# Tensor Compiler

The goal of the project is a full-fledged tensor compiler.
Currently implemented:
- import ONNX model into an internal compute graph (`GraphBuilder`)
- validate graph consistency (`verify`)
- export graph to Graphviz `.dot` for visualization
- lower graph to MLIR / LLVM IR / assembly / object file

## How to Install

```bash
git clone git@github.com:daniilgriga/TensorCompiler.git
cd TensorCompiler/
```

## Dependencies

### Frontend-only build

- CMake
- Protobuf

### Codegen build (`BUILD_CODEGEN=ON`)

- LLVM + MLIR (project is developed against LLVM/MLIR 22)

macOS (Homebrew):

```bash
brew install cmake ninja googletest protobuf llvm
```

Ubuntu 24.04 (example):

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build libgtest-dev protobuf-compiler libprotobuf-dev wget gnupg
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null
echo "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-22 main" | sudo tee /etc/apt/sources.list.d/llvm.list
sudo apt-get update
sudo apt-get install -y llvm-22-dev libmlir-22-dev mlir-22-tools
```

## How to Build

### Frontend-only

```bash
# Debug + tests
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
cmake --build build/debug

# Release
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
cmake --build build/release
```

### Codegen (MLIR/LLVM enabled)

```bash
cmake -S . -B build-codegen \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTS=ON \
  -DBUILD_CODEGEN=ON

cmake --build build-codegen
```

If LLVM/MLIR are installed in a non-default location (for example Homebrew on macOS):

```bash
cmake -S . -B build-codegen \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTS=ON \
  -DBUILD_CODEGEN=ON \
  -DLLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir
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

### Emit MLIR

```bash
./build-codegen/tc_main tests/models/add.onnx --emit-mlir
```

### Emit LLVM IR

```bash
./build-codegen/tc_main tests/models/add.onnx --emit-llvm
./build-codegen/tc_main tests/models/add.onnx --emit-llvm -o output/add.ll
```

### Emit Assembly

```bash
./build-codegen/tc_main tests/models/add.onnx --emit-asm
./build-codegen/tc_main tests/models/add.onnx --emit-asm -o output/add.s
```

### Emit Object File

```bash
./build-codegen/tc_main tests/models/add.onnx --emit-obj
./build-codegen/tc_main tests/models/add.onnx --emit-obj -o output/add.o
```

### Target + Optimization Level

```bash
./build-codegen/tc_main tests/models/add.onnx --emit-asm --target=x86_64-unknown-linux-gnu -O2 -o output/add.s
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

### Generate Test Models

```bash
python3 -m pip install numpy onnx
python3 tests/gen_test_models.py
```

### Build Tests (frontend)

```bash
cmake -S . -B build/tests -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
cmake --build build/tests
```

### Run Tests (frontend)

```bash
ctest --test-dir build/tests --output-on-failure
```

### Build + Run Tests (codegen)

```bash
cmake -S . -B build-codegen -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON -DBUILD_CODEGEN=ON
cmake --build build-codegen
ctest --test-dir build-codegen --output-on-failure
```
