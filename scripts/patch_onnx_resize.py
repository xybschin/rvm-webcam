#!/usr/bin/env python3
"""Patch RVM ONNX models for MIGraphX compatibility.

Patches:
  1. Resize: mode='linear' → 'nearest' for dynamic scales/sizes inputs
  2. AveragePool: add explicit count_include_pad=0 (MIGraphX rejects dynamic
     shape AveragePool with the default count_include_pad=1)
"""

import sys
from pathlib import Path

import onnx
from onnx import helper


def _is_constant(model, tensor_name):
    for init in model.graph.initializer:
        if init.name == tensor_name:
            return True
    return False


def patch_model(model_path, out_path):
    model = onnx.load(model_path)
    patches = 0

    for node in model.graph.node:
        if node.op_type == "Resize":
            mode_attr = None
            for attr in node.attribute:
                if attr.name == "mode":
                    mode_attr = attr
                    break
            if mode_attr is None or mode_attr.s != b"linear":
                continue
            has_dynamic = False
            if len(node.input) >= 3:
                if not _is_constant(model, node.input[2]):
                    has_dynamic = True
            if len(node.input) >= 4 and node.input[3]:
                if not _is_constant(model, node.input[3]):
                    has_dynamic = True
            if has_dynamic:
                mode_attr.s = b"nearest"
                patches += 1
                print(f"  {node.name}: Resize mode linear → nearest", flush=True)

        elif node.op_type == "AveragePool":
            has_count = any(a.name == "count_include_pad" for a in node.attribute)
            if not has_count:
                node.attribute.append(
                    helper.make_attribute("count_include_pad", 0)
                )
                patches += 1
                print(f"  {node.name}: added count_include_pad=0", flush=True)
            else:
                for attr in node.attribute:
                    if attr.name == "count_include_pad" and attr.i == 1:
                        attr.i = 0
                        patches += 1
                        print(f"  {node.name}: count_include_pad 1→0",
                              flush=True)

    onnx.save(model, out_path)
    return patches


def main():
    if len(sys.argv) > 1:
        models_dir = Path(sys.argv[1])
    else:
        models_dir = Path(__file__).resolve().parent.parent / "models"

    models = sorted(models_dir.glob("rvm_*.onnx"))
    models = [m for m in models if ".patched" not in m.name]

    if not models:
        print("No rvm_*.onnx models found in", models_dir)
        sys.exit(1)

    for model_path in models:
        stem = model_path.stem
        out_path = model_path.with_name(f"{stem}.patched.onnx")
        print(f"Processing: {model_path.name} → {out_path.name}")
        n = patch_model(str(model_path), str(out_path))
        print(f"  {n} patch(es) applied\n")


if __name__ == "__main__":
    main()
