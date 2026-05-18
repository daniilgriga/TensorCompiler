#!/usr/bin/env python3
"""
End-to-end correctness check.

Usage:
    python3 verify.py <model.onnx> [N C H W]

Compares TensorCompiler JIT output with onnxruntime reference.
"""

import sys
import os
import subprocess
import struct
import tempfile
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("error: onnxruntime not installed. Run: pip install onnxruntime")
    sys.exit(1)


def get_model_io_info(model_path):
    """Return first input name/shape and output total elements."""
    sess = ort.InferenceSession(model_path)
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    return inp.name, inp.shape, out.shape


def run_onnxruntime(model_path, input_data):
    sess = ort.InferenceSession(model_path)
    inp_name = sess.get_inputs()[0].name
    result = sess.run(None, {inp_name: input_data})
    return result[0].flatten()


def run_tc(tc_bin, model_path, input_data, shape, out_elems):
    N, C, H, W = shape
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        input_data.astype(np.float32).tofile(f)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [
                tc_bin, model_path,
                "--run",
                "--input", tmp_path,
                "--in-N", str(N),
                "--in-C", str(C),
                "--in-H", str(H),
                "--in-W", str(W),
                "--out-elems", str(out_elems),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("error: tc_main failed:")
            print(result.stderr)
            return None
        return np.array([float(x) for x in result.stdout.strip().split()])
    finally:
        os.unlink(tmp_path)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.dirname(script_dir)

    if len(sys.argv) < 2:
        print("usage: python3 verify.py <model.onnx> [N C H W]")
        sys.exit(1)
    model_path = sys.argv[1]

    # find tc_main binary
    for candidate in ("build-codegen", "build"):
        tc_bin = os.path.join(repo_root, candidate, "tc_main")
        if os.path.exists(tc_bin):
            break
    else:
        print("error: tc_main not found in build-codegen/ or build/")
        print("Build with: cmake --build build")
        sys.exit(1)

    # get model info
    inp_name, inp_shape, out_shape = get_model_io_info(model_path)
    inp_shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp_shape]
    N, C, H, W = inp_shape
    out_elems = 1
    for d in out_shape:
        if isinstance(d, int) and d > 0:
            out_elems *= d

    # override shape from CLI if provided
    if len(sys.argv) == 6:
        N, C, H, W = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

    print(f"model:      {model_path}")
    print(f"input:      {inp_name}  shape=[{N},{C},{H},{W}]")
    print(f"out_elems:  {out_elems}")
    print()

    # random input
    rng = np.random.default_rng(42)
    input_data = rng.random((N, C, H, W), dtype=np.float32)

    # --- reference ---
    ref = run_onnxruntime(model_path, input_data)
    print(f"onnxruntime: {ref}")

    # --- our compiler ---
    tc = run_tc(tc_bin, model_path, input_data, (N, C, H, W), out_elems)
    if tc is None:
        sys.exit(1)
    print(f"tc_main:     {tc}")

    # --- compare ---
    max_diff = float(np.max(np.abs(ref - tc)))
    print()
    print(f"max abs diff: {max_diff:.2e}")

    tol = 1e-4
    if max_diff <= tol:
        print(f"PASS  (tolerance {tol})")
    else:
        print(f"FAIL  (tolerance {tol})")
        sys.exit(1)


if __name__ == "__main__":
    main()
