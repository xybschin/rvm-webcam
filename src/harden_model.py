import onnx
import numpy as np
from onnx import helper, TensorProto


def harden_downsample_ratio(model_path: str, ratio: float) -> bytes:
    """Read an RVM ONNX model and return a serialized version where
    *downsample_ratio* has been converted from a dynamic input to a
    compile-time constant ``[1.0, 1.0, ratio, ratio]`` fed directly into
    the Resize node.

    This allows MIGraphXExecutionProvider to accept the graph — it
    rejects ``Resize(linear)`` when the scale is not compile-time known.

    If the model does not contain a ``downsample_ratio`` input (e.g., it
    was already hardened), the original bytes are returned unchanged.
    """
    model = onnx.load(model_path)

    # ── check if downsample_ratio is already a constant ─────────────
    input_names = {i.name for i in model.graph.input}
    if "downsample_ratio" not in input_names:
        return model.SerializeToString()

    # ── locate the subgraph we need to replace ──────────────────────
    #   Constant_1 → 387 ─┐
    #                      ├─ Concat_2(axis=0) → 388 → Resize_3
    #   downsample_ratio ──┘
    concat_idx = None
    for i, node in enumerate(model.graph.node):
        if node.name == "Concat_2":
            concat_idx = i
            break

    if concat_idx is None:
        raise RuntimeError("Could not locate Concat_2 in the ONNX graph")

    # ── build the constant scale tensor ─────────────────────────────
    scale = np.array([1.0, 1.0, ratio, ratio], dtype=np.float32)

    new_constant = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["388"],
        name="HardenedScale",
        value=helper.make_tensor(
            "hardened_scale_value",
            TensorProto.FLOAT,
            scale.shape,
            scale.tobytes(),
            raw=True,
        ),
    )

    # ── replace Concat_2 with the new Constant node ─────────────────
    model.graph.node.remove(model.graph.node[concat_idx])
    model.graph.node.insert(concat_idx, new_constant)

    # ── remove downsample_ratio from graph inputs ───────────────────
    for i, inp in enumerate(model.graph.input):
        if inp.name == "downsample_ratio":
            del model.graph.input[i]
            break

    return model.SerializeToString()
