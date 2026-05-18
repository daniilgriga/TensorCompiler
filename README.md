![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![macOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0)
![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-%23008FBA.svg?style=for-the-badge&logo=cmake&logoColor=white)
![MLIR](https://img.shields.io/badge/MLIR%2FLLVM-22-blue?style=for-the-badge)

# Tensor Compiler

A tensor compiler that takes ONNX models, lowers them through MLIR to LLVM IR,
and either emits native code (object file / shared library / assembly) or
JIT-executes the model with real input data.

**Supported ONNX ops:** Add, Mul, Conv, Relu, MatMul, Gemm, Reshape, Flatten,
MaxPool, GlobalAveragePool, BatchNormalization, Softmax

End-to-end verified against onnxruntime — including **ResNet18**
(max abs diff `5.25e-06`).

---

## Part 1 — Frontend (ONNX → Compute Graph)

### Dependencies

- CMake ≥ 3.28
- Protobuf

macOS:
```bash
brew install cmake protobuf googletest
```

Ubuntu 24.04:
```bash
sudo apt-get install -y cmake libgtest-dev protobuf-compiler libprotobuf-dev
```

### Build

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build
```

### Usage

Print graph statistics:
```bash
./build/tc_main tests/models/conv_relu_gemm.onnx
```

Export compute graph to Graphviz DOT:
```bash
./build/tc_main tests/models/conv_relu_gemm.onnx --dot output/conv_relu_gemm.dot
dot -Tsvg output/conv_relu_gemm.dot -o output/conv_relu_gemm.svg
```

### Tests

```bash
ctest --test-dir build --output-on-failure
```

### Graph Visualizations

The repo includes sample visualizations:

- `output/conv_relu_gemm.svg` — small test model (Conv → Relu → Reshape → Gemm):

<p align="center">
  <img src="output/conv_relu_gemm.svg" alt="conv_relu_gemm graph">
</p>

- `output/adv_inception_v3.svg` — [Adversarial Inception v3](https://github.com/onnx/models) from the ONNX Model Zoo: [view here](https://github.com/daniilgriga/TensorCompiler/blob/dev/output/adv_inception_v3.svg)

---

## Part 2 — Codegen (MLIR / LLVM IR / Assembly)

### Additional Dependencies

- LLVM + MLIR 22

macOS (Homebrew):
```bash
brew install llvm
```

Ubuntu 24.04:
```bash
sudo apt-get install -y wget gnupg
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null
echo "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-22 main" | sudo tee /etc/apt/sources.list.d/llvm.list
sudo apt-get update
sudo apt-get install -y llvm-22-dev libmlir-22-dev mlir-22-tools
```

### Build

```bash
cmake -S . -B build \
  -DBUILD_CODEGEN=ON \
  -DBUILD_TESTS=ON

cmake --build build
```

On macOS with Homebrew LLVM (non-default location):
```bash
cmake -S . -B build \
  -DBUILD_CODEGEN=ON \
  -DBUILD_TESTS=ON \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
  -DLLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm \
  -DMLIR_DIR=/opt/homebrew/opt/llvm/lib/cmake/mlir

cmake --build build
```

### Emit MLIR

```bash
./build/tc_main tests/models/add.onnx --emit-mlir
```

### Emit LLVM IR

```bash
./build/tc_main tests/models/add.onnx --emit-llvm
./build/tc_main tests/models/add.onnx --emit-llvm -o output/add.ll
```

### Emit Assembly

```bash
./build/tc_main tests/models/add.onnx --emit-asm
./build/tc_main tests/models/add.onnx --emit-asm > output/add.s
```

### Emit Object File

```bash
./build/tc_main tests/models/add.onnx --emit-obj -o output/add.o
```

### Emit Shared Library (AOT)

```bash
./build/tc_main tests/models/conv_relu_gemm.onnx --emit-so -o model.so
```

### Optimization Levels

```bash
./build/tc_main tests/models/add.onnx --emit-obj -O2 -o output/add.o
```

Supported: `-O0` (default), `-O1`, `-O2`, `-O3`.

### Cross-compilation

```bash
./build/tc_main tests/models/add.onnx --emit-asm \
  --target=x86_64-unknown-linux-gnu \
  -O2 -o output/add.s
```

### Tests

```bash
ctest --test-dir build --output-on-failure
```

---

## Part 3 — Execution (JIT + Verification)

### Run a Model (JIT)

JIT-compile and execute with real input:

```bash
# prepare input: raw float32 binary, shape [1, 1, 5, 5]
python3 -c "import numpy as np; np.ones((1,1,5,5), dtype=np.float32).tofile('input.bin')"

./build/tc_main tests/models/conv_relu_gemm.onnx \
    --run --input input.bin \
    --in-N 1 --in-C 1 --in-H 5 --in-W 5 \
    --out-elems 10
```

### Verify Correctness vs onnxruntime

```bash
pip install onnxruntime numpy
python3 tests/verify.py <model.onnx> [N C H W]
```

Example — small test model:
```bash
python3 tests/verify.py tests/models/conv_relu_gemm.onnx
```
```
model:       tests/models/conv_relu_gemm.onnx
input:       X  shape=[1,1,5,5]
out_elems:   10

onnxruntime: [46.003456 46.003456 ...]
tc_main:     [46.00346  46.00346  ...]

max abs diff: 3.88e-06
PASS  (tolerance 0.0001)
```

Example — ResNet18:
```bash
python3 tests/verify.py resnet18.onnx
```
```
model:       resnet18.onnx
input:       X  shape=[1,3,224,224]
out_elems:   1000

max abs diff: 5.25e-06
PASS  (tolerance 0.0001)
```

### Generate Test Models

```bash
pip install numpy onnx
python3 tests/gen_test_models.py
```
