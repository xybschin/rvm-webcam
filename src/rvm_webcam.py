#!/usr/bin/env python3
"""
High-Performance RVM Virtual Camera (ROCm 6+ / MIGraphX Zero-Copy Engine)
Optimized for AMD Radeon RX 9070 XT (RDNA 4)
"""

import copy
import ctypes
import ctypes.util
import dataclasses
import fcntl
import json
import math
import mmap
import os
import queue
import select
import shutil
import signal
import struct
import subprocess
import threading
import time
from pathlib import Path

import onnx
from onnx import helper, TensorProto

import click
import numpy as np
import onnxruntime as ort


class WorkerException(Exception):
    """Encapsulates exceptions thrown inside background worker threads."""

    pass


class LoopbackConsumerMonitor:
    """Tracks whether a real consumer is reading from a v4l2loopback
    CAPTURE device, so the GPU pipeline only runs (and the physical
    camera's "shutter"/LED activates) while something is actually
    watching the virtual camera.

    v4l2loopback fires a private V4L2 event, ``V4L2_EVENT_PRI_CLIENT_USAGE``
    (id 0x10e00001), whenever a CAPTURE-side opener calls
    ``VIDIOC_STREAMON``/``VIDIOC_STREAMOFF``. The event payload's first
    4 bytes are a ``u32 count`` computed by the driver as
    ``!has_capture_token(stream_tokens)``, where ``stream_tokens`` bits
    are *set* when a token is free/unclaimed. So ``count == 0`` means
    "no consumer is streaming" (capture token still available) and
    ``count != 0`` means "a consumer has acquired the capture token and
    is actively streaming". This is the same mechanism browsers/OBS use
    to show a "camera in use" indicator only when a page/tab is actually
    reading frames, but pyvirtualcam's Linux backend (a bare ``write()`` to the
    device) does not expose it, so this reimplements it via raw ioctls.

    Falls back to permanently "active" if the ioctl/event subscription
    is unavailable (e.g. non-Linux, non-v4l2loopback backend, or an
    older v4l2loopback version without this event), so pipelines never
    silently stop producing frames on unsupported platforms.
    """

    _VIDIOC_SUBSCRIBE_EVENT = 0x4020565A
    _VIDIOC_DQEVENT = 0x80885659
    _V4L2_EVENT_PRI_CLIENT_USAGE = 0x08000000 + 0x08E00000 + 1
    _V4L2_EVENT_SUB_FL_SEND_INITIAL = 1

    def __init__(self, device: str):
        self.device = device
        self._fd: int | None = None
        self._supported = False
        self._active = True  # fail-open if monitoring isn't supported
        self._lock = threading.Lock()
        self._stopped = False
        self._thread: threading.Thread | None = None

        try:
            self._fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
            sub = struct.pack(
                "=III5I",
                self._V4L2_EVENT_PRI_CLIENT_USAGE,
                0,
                self._V4L2_EVENT_SUB_FL_SEND_INITIAL,
                0,
                0,
                0,
                0,
                0,
            )
            fcntl.ioctl(self._fd, self._VIDIOC_SUBSCRIBE_EVENT, sub)
            self._supported = True
            self._active = False  # will be set True by the initial event if a consumer exists
        except OSError as e:
            click.echo(
                f"[rvm-webcam] Warning: consumer detection unavailable on "
                f"{device} ({e}); pipeline will always run.",
                err=True,
            )
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        assert self._fd is not None
        while not self._stopped:
            try:
                r, _, x = select.select([self._fd], [], [self._fd], 0.5)
            except OSError:
                break
            if not r and not x:
                continue
            while True:
                buf = bytearray(136)
                try:
                    fcntl.ioctl(self._fd, self._VIDIOC_DQEVENT, buf)
                except OSError:
                    break
                count = struct.unpack_from("=I", bytes(buf), 8)[0]
                with self._lock:
                    self._active = count != 0

    @property
    def is_active(self) -> bool:
        """True if a consumer is currently streaming from the device
        (or if consumer detection is unsupported and we fail open)."""
        if not self._supported:
            return True
        with self._lock:
            return self._active

    def stop(self):
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None


def _load_hip_runtime() -> ctypes.CDLL | None:
    """Resolve libamdhip64.so with multi-path fallback.

    Priority:
      1. ctypes.CDLL('libamdhip64.so') -- relies on LD_LIBRARY_PATH / ld cache
      2. ctypes.util.find_library('amdhip64')
      3. Explicit well-known ROCm installation paths
    """
    # Strategy 1: default loader
    try:
        return ctypes.CDLL("libamdhip64.so")
    except OSError:
        pass

    # Strategy 2: ctypes.util.find_library
    lib_path = ctypes.util.find_library("amdhip64")
    if lib_path is not None:
        try:
            return ctypes.CDLL(lib_path)
        except OSError:
            pass

    # Strategy 3: well-known system paths
    candidates = [
        "/opt/rocm/lib/libamdhip64.so",
        "/opt/rocm/hip/lib/libamdhip64.so",
        "/opt/rocm-6.0.0/lib/libamdhip64.so",
        "/opt/rocm-6.1.0/lib/libamdhip64.so",
        "/opt/rocm-6.2.0/lib/libamdhip64.so",
        "/opt/rocm-6.3.0/lib/libamdhip64.so",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            try:
                return ctypes.CDLL(candidate)
            except OSError:
                continue

    return None


@dataclasses.dataclass
class PinnedBuffer:
    """Encapsulates a NumPy array backed by POSIX mmap and registered via HIP DMA."""

    array: np.ndarray
    buf: mmap.mmap
    addr: int
    bytes_needed: int
    is_pinned: bool

    def cleanup(self):
        """Unregisters memory from the HIP driver and releases physical locks."""
        if self.is_pinned:
            try:
                hip = _load_hip_runtime()
                if hip is not None:
                    rc = hip.hipHostUnregister(ctypes.c_void_p(self.addr))
                    if rc != 0:
                        click.echo(
                            f"[rvm-webcam] Warning: hipHostUnregister failed (rc={rc})",
                            err=True,
                        )
            except Exception as e:
                click.echo(
                    f"[rvm-webcam] Warning: Error unloading HIP driver handle: {e}",
                    err=True,
                )
            self.is_pinned = False

        try:
            libc = ctypes.CDLL(None)
            libc.munlock(ctypes.c_void_p(self.addr), ctypes.c_size_t(self.bytes_needed))
        except Exception:
            pass

        try:
            self.buf.close()
        except Exception:
            pass


def allocate_pinned_numpy(shape: tuple, dtype: np.dtype) -> PinnedBuffer:
    """Allocates page-locked host memory registered with ROCm via hipHostRegister.

    Guarantees DMA access across PCIe without driver staging copies.
    """
    dtype = np.dtype(dtype)
    bytes_needed = int(np.prod(shape) * dtype.itemsize)

    # 1. POSIX Anonymous Shared Memory Mapping
    buf = mmap.mmap(-1, bytes_needed, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS)

    # 2. Extract Base Physical/Virtual Memory Address
    c_array = (ctypes.c_char * bytes_needed).from_buffer(buf)
    addr = ctypes.addressof(c_array)

    # 3. Apply POSIX page-lock to avoid swapping
    try:
        libc = ctypes.CDLL(None)
        libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(bytes_needed))
    except Exception as e:
        click.echo(f"[rvm-webcam] Warning: mlock failed: {e}", err=True)

    # 4. Register with HIP Runtime for Hardware DMA Access
    is_pinned = False
    try:
        hip = _load_hip_runtime()
        if hip is not None:
            rc = hip.hipHostRegister(
                ctypes.c_void_p(addr), ctypes.c_size_t(bytes_needed), 0
            )
            if rc == 0:
                is_pinned = True
            else:
                click.echo(
                    f"[rvm-webcam] Warning: hipHostRegister failed (rc={rc}). Staging copies will occur.",
                    err=True,
                )
        else:
            click.echo(
                "[rvm-webcam] Warning: libamdhip64.so not found via any resolution strategy. Hardware pinning disabled.",
                err=True,
            )
    except OSError as e:
        click.echo(
            f"[rvm-webcam] Warning: Could not load libamdhip64.so ({e}). Hardware pinning disabled.",
            err=True,
        )

    # 5. Create Zero-Copy NumPy View
    array = np.frombuffer(buf, dtype=dtype).reshape(shape)
    if not array.flags.writeable:
        array.setflags(write=True)

    return PinnedBuffer(
        array=array,
        buf=buf,
        addr=addr,
        bytes_needed=bytes_needed,
        is_pinned=is_pinned,
    )


class FrameGrabber:
    """Continuous background thread draining the V4L2 device buffer queue
    via an ffmpeg subprocess pipe."""

    def __init__(self, read_fn, error_q: queue.Queue):
        self._read_fn = read_fn
        self.error_q = error_q
        self._cond = threading.Condition()
        self._frame = None
        self._new = False
        self._stopped = False
        self._thread = threading.Thread(target=self._run_guarded, daemon=True)
        self._thread.start()

    def _run_guarded(self):
        try:
            self._run()
        except Exception as e:
            self.error_q.put(WorkerException(f"FrameGrabber failed: {e}"))

    def _run(self):
        while not self._stopped:
            frame = self._read_fn()
            if frame is None:
                time.sleep(0.001)
                continue
            with self._cond:
                self._frame = frame
                self._new = True
                self._cond.notify_all()

    def wait_new(self, timeout=1.0) -> np.ndarray | None:
        with self._cond:
            if not self._new:
                if not self._cond.wait(timeout=timeout):
                    return None
            self._new = False
            return self._frame

    def stop(self):
        self._stopped = True
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)


class FramePrep:
    """Zero-allocation pipeline: FP32 host scaling, FP16 cast at the GPU input boundary."""

    def __init__(
        self, grabber: FrameGrabber, width: int, height: int, error_q: queue.Queue
    ):
        self.grabber = grabber
        self.W = width
        self.H = height
        self.error_q = error_q

        # Pinned host memory input buffers mapped directly into FP16 layout
        self.pinned_bufs = [
            allocate_pinned_numpy((1, 3, height, width), dtype=np.float16),
            allocate_pinned_numpy((1, 3, height, width), dtype=np.float16),
        ]
        self.bufs = [pb.array for pb in self.pinned_bufs]

        # FP32 scratch buffer prevents CPU float16 SIMD execution bottlenecks
        self.pinned_scratch = allocate_pinned_numpy(
            (height, width, 3), dtype=np.float32
        )
        self._hwc_scratch = self.pinned_scratch.array

        self._cond = threading.Condition()
        self._ready_idx = None
        self._inuse_idx = None
        self._stopped = False
        self._thread = threading.Thread(target=self._run_guarded, daemon=True)
        self._thread.start()

    def _free_idx(self):
        for i in (0, 1):
            if i != self._ready_idx and i != self._inuse_idx:
                return i
        return None

    def _run_guarded(self):
        try:
            self._run()
        except Exception as e:
            self.error_q.put(WorkerException(f"FramePrep failed: {e}"))

    def _run(self):
        scale = np.float32(1.0 / 255.0)
        while not self._stopped:
            frame = self.grabber.wait_new(timeout=1.0)
            if frame is None:
                continue

            rgb = frame
            if frame.shape[2] == 4:
                rgb = frame[..., :3]
            if frame.shape[0] != self.H or frame.shape[1] != self.W:
                from PIL import Image

                rgb = np.array(Image.fromarray(rgb).resize((self.W, self.H)))

            with self._cond:
                fi = self._free_idx()
                if fi is None:
                    self._cond.wait(timeout=1.0)
                    continue

                # FP32 Vectorized Scaling Operation
                np.multiply(rgb, scale, out=self._hwc_scratch, casting="unsafe")

                # Single-pass FP32 -> FP16 downcast + transpose into DMA host target buffer
                np.copyto(self.bufs[fi][0], self._hwc_scratch.transpose(2, 0, 1))

                self._ready_idx = fi
                self._cond.notify_all()

    def take(self, timeout=2.0) -> tuple[int | None, np.ndarray | None]:
        with self._cond:
            while self._ready_idx is None and not self._stopped:
                if not self._cond.wait(timeout=timeout):
                    if self._ready_idx is None:
                        return None, None
            if self._stopped or self._ready_idx is None:
                return None, None
            idx = self._ready_idx
            self._ready_idx = None
            self._inuse_idx = idx
            return idx, self.bufs[idx]

    def release(self, idx: int):
        with self._cond:
            if self._inuse_idx == idx:
                self._inuse_idx = None
            self._cond.notify_all()

    def stop(self):
        self._stopped = True
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)

    def cleanup(self):
        self.stop()
        for pb in self.pinned_bufs:
            pb.cleanup()
        self.pinned_scratch.cleanup()


class RecurrentStateBufferGPU:
    """Manages RVM r1-r4 recurrent state tensors.

    Initial states are allocated at their *final* static spatial shape
    instead of a broadcastable (1, C, 1, 1) shape.

    MIGraphX compiles a fixed computation graph for the exact tensor
    shapes seen on the first inference call. If the r*i input shape
    changed between frame 1 (broadcast shape) and frame 2 (real spatial
    shape fed back from r*o), MIGraphX would have to recompile the
    entire graph from scratch on frame 2 -- which can hang for minutes
    and blocks the main thread inside a native call, making the process
    appear stuck and unresponsive to Ctrl-C. Keeping the shape constant
    across every call from frame 1 onward avoids any recompilation.

    Shape derivation (empirically verified against the RVM ONNX graph):
    RVM first resizes ``src`` by ``downsample_ratio`` (ceil rounding),
    then the encoder halves the spatial resolution (ceil rounding) once
    per recurrent stage: r1 after 1 halving, r2 after 2, r3 after 3,
    r4 after 4.
    """

    def __init__(
        self,
        height: int,
        width: int,
        downsample_ratio: float = 0.25,
        device_id: int = 0,
    ):
        self.device_id = device_id
        self.channels = {"r1": 16, "r2": 20, "r3": 40, "r4": 64}
        self.num_halvings = {"r1": 1, "r2": 2, "r3": 3, "r4": 4}

        h0 = math.ceil(height * downsample_ratio)
        w0 = math.ceil(width * downsample_ratio)

        def halved(n: int) -> tuple[int, int]:
            h, w = h0, w0
            for _ in range(n):
                h = math.ceil(h / 2)
                w = math.ceil(w / 2)
            return h, w

        self._states = {
            name: np.zeros((1, C, *halved(self.num_halvings[name])), dtype=np.float16)
            for name, C in self.channels.items()
        }

        self.device_type = self._detect_device_type(device_id)

    def _detect_device_type(self, device_id: int) -> str:
        """Return a device identifier usable by OrtValue (``'cpu'`` fallback)."""
        available_providers = ort.get_available_providers()
        click.echo(f"[rvm-webcam] Available Execution Providers: {available_providers}")

        required = ["MIGraphXExecutionProvider"]
        if not any(p in available_providers for p in required):
            raise RuntimeError(
                f"Execution provider assertion failed. None of {required} are present "
                f"in current ONNX Runtime build. Available providers: {available_providers}"
            )

        dummy = np.zeros((1,), dtype=np.float32)
        for candidate in ["rocm", "cuda", "hip"]:
            try:
                handle = ort.OrtValue.ortvalue_from_numpy(dummy, candidate, device_id)
                del handle
                click.echo(f"[rvm-webcam] Confirmed ORT device identifier: '{candidate}'")
                return candidate
            except Exception:
                pass

        return "cpu"

    def bind_inputs(self, io_binding: ort.IOBinding):
        """Bind state input tensors (r1i, r2i, r3i, r4i) as CPU inputs.

        Using ``bind_cpu_input`` rather than ``bind_ortvalue_input`` so
        that ORT handles device transfer to the MIGraphX EP correctly.
        """
        for name in ["r1", "r2", "r3", "r4"]:
            io_binding.bind_cpu_input(f"{name}i", self._states[name])

    def store_outputs(self, ortvals: list[ort.OrtValue]):
        """Store the four output state tensors (r1o, r2o, r3o, r4o) as the
        next frame's inputs.  Call *after* ``run_with_iobinding``.

        Data is **copied** out of the ORT-managed output buffers to prevent
        memory corruption from internal buffer reuse on the next run.
        """
        for i, name in enumerate(["r1", "r2", "r3", "r4"]):
            self._states[name] = ortvals[i].numpy().copy()


class MIGraphXSession:
    """ONNX Runtime session wrapper targeting ROCm / MIGraphX execution graph."""

    def __init__(
        self,
        model_path: str,
        height: int,
        width: int,
        downsample_ratio: float = 0.25,
        device_id: int = 0,
        cache_dir: str | None = None,
    ):
        ort.set_default_logger_severity(3)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        model_bytes = self._harden_model(model_path, downsample_ratio)

        if "MIGraphXExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "MIGraphX execution provider not available. "
                "Install onnxruntime-rocm or build with MIGraphX support."
            )
        providers = ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            providers = [
                (
                    "MIGraphXExecutionProvider",
                    {
                        "device_id": str(device_id),
                        "migraphx_fp16_enable": "1",
                        "migraphx_model_cache_dir": str(cache_path),
                    },
                ),
                "CPUExecutionProvider",
            ]

        click.echo(f"[rvm-webcam] Instantiating acceleration engine for: {model_path}")
        self.session = ort.InferenceSession(
            model_bytes, so, providers=providers
        )

        # Confirm acceleration providers bound successfully
        active_providers = self.session.get_providers()
        click.echo(f"[rvm-webcam] Active Session Providers: {active_providers}")

        self.io_binding = self.session.io_binding()
        self._validate_model_signature(height, width)

    def _validate_model_signature(self, expected_height: int, expected_width: int):
        """Assert src input dtype and static spatial dimensions."""
        inputs_by_name = {inp.name: inp for inp in self.session.get_inputs()}

        for name, exp_type, exp_shape in [
            ("src", "tensor(float16)", (1, 3, expected_height, expected_width)),
        ]:
            actual = inputs_by_name[name]
            if actual.type != exp_type:
                raise RuntimeError(
                    f"ONNX graph type assertion failed for input '{name}': "
                    f"Model expects '{actual.type}', but runtime pipeline requires '{exp_type}'."
                )
            for axis_idx, (exp_dim, act_dim) in enumerate(zip(exp_shape, actual.shape)):
                if isinstance(act_dim, int) and act_dim != exp_dim:
                    raise RuntimeError(
                        f"ONNX graph shape mismatch for input '{name}' at axis {axis_idx}: "
                        f"Model static shape requires {act_dim}, runtime configured for {exp_dim}."
                    )
            click.echo(
                f"[rvm-webcam] Verified input signature '{name}': dtype={actual.type}, shape={actual.shape}"
            )

        outputs_by_name = {o.name: o for o in self.session.get_outputs()}
        for name, exp_type in [("fgr_fp32", "tensor(float)"), ("pha_fp32", "tensor(float)")]:
            actual = outputs_by_name.get(name)
            if actual is None:
                click.echo(
                    f"[rvm-webcam] WARNING: expected cast output '{name}' not found "
                    f"in model (original model without FP32 cast?)"
                )
                continue
            if actual.type != exp_type:
                raise RuntimeError(
                    f"ONNX graph type assertion failed for output '{name}': "
                    f"expected '{exp_type}', got '{actual.type}'"
                )

    def _harden_model(self, model_path: str, ratio: float) -> bytes:
        """Replace the dynamic *downsample_ratio* input with a compile-time
        constant so that MIGraphXExecutionProvider accepts the Resize(linear) node.

        Returns serialized model bytes on success, or the original *model_path*
        string (for ORT to load from disk) if the file is not found or the
        ONNX graph doesn't have the expected structure.
        """
        try:
            model = onnx.load(model_path)
        except Exception as exc:
            click.echo(
                f"[rvm-webcam] WARNING: could not load model for hardening ({exc}); "
                "loading original"
            )
            return model_path  # type: ignore[return-value]  # ORT accepts str path or bytes

        input_names = {i.name for i in model.graph.input}
        if "downsample_ratio" not in input_names:
            return model.SerializeToString()

        scale = np.array([1.0, 1.0, ratio, ratio], dtype=np.float32)

        to_remove = []
        insertion_point = None
        for i, node in enumerate(model.graph.node):
            if node.name in ("Constant_1", "Concat_2"):
                to_remove.append(node)
            if node.name == "Resize_3":
                insertion_point = i

        if insertion_point is None or not to_remove:
            return model.SerializeToString()

        for node in to_remove:
            model.graph.node.remove(node)

        # Adjust insertion point for removed nodes before Resize_3
        for node in to_remove:
            idx = None
            for j, n in enumerate(model.graph.node):
                if n.name == node.name:
                    idx = j
                    break
            if idx is not None and idx < insertion_point:
                insertion_point -= 1

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

        model.graph.node.insert(insertion_point, new_constant)

        for i, inp in enumerate(model.graph.input):
            if inp.name == "downsample_ratio":
                del model.graph.input[i]
                break

        click.echo(f"[rvm-webcam] Hardened downsample_ratio={ratio} into model")

        self._cast_model_outputs_to_fp32(model)

        return model.SerializeToString()

    @staticmethod
    def _cast_model_outputs_to_fp32(model: onnx.ModelProto):
        """Insert Cast(FP32) nodes for fgr and pha outputs so MIGraphX
        converts FP16→FP32 on GPU; CPU composition avoids slow x86 FP16 ops."""
        cast_targets = {"fgr", "pha"}
        output_by_name: dict[str, onnx.ValueInfoProto] = {
            o.name: o for o in model.graph.output
        }

        for target in cast_targets & output_by_name.keys():
            o = output_by_name[target]
            new_name = f"{target}_fp32"
            cast_node = helper.make_node(
                "Cast",
                inputs=[target],
                outputs=[new_name],
                name=f"Cast_{target}_to_fp32",
                to=TensorProto.FLOAT,
            )
            model.graph.node.append(cast_node)

            # Deep-copy output info so symbolic dims (batch, height, width)
            # are preserved; only change name and dtype.
            new_output = copy.deepcopy(o)
            new_output.name = new_name
            new_output.type.tensor_type.elem_type = TensorProto.FLOAT
            # Replace in-place to preserve output order
            idx = list(model.graph.output).index(o)
            model.graph.output.remove(o)
            model.graph.output.insert(idx, new_output)


DEFAULTS = {
    "model_path": "rvm_mobilenetv3_fp16.onnx",
    "input_device": "/dev/video0",
    "output_device": "/dev/video10",
    "width": 1280,
    "height": 720,
    "fps": 30,
    "downsample_ratio": 0.25,
    "device_id": 0,
    "cache_dir": None,
    "bg_color": "0,255,0",
    "bg_image": None,
}


def _load_config():
    for config_path in [
        Path.home() / ".config" / "rvm-webcam" / "config.json",
        Path("/etc/rvm-webcam/config.json"),
    ]:
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
    return {}


def _load_background(
    bg_image: str | None, bg_color: str | None, width: int, height: int
) -> np.ndarray:
    """Return the composite background as a float32 array.

    If *bg_image* is given, returns a full ``(height, width, 3)`` RGB
    array (resized to the output resolution). Otherwise parses
    *bg_color* as an ``"R,G,B"`` string and returns a broadcastable
    ``(3,)`` array -- the compositor's ``bg * (1 - pha)`` term
    broadcasts either shape correctly against the ``(height, width, 3)``
    foreground/alpha.
    """
    if bg_image:
        from PIL import Image

        try:
            img = Image.open(bg_image).convert("RGB").resize((width, height))
        except Exception as exc:
            raise click.BadParameter(
                f"Could not load image: {bg_image} ({exc})", param_hint="--bg-image"
            )
        click.echo(f"[rvm-webcam] Using background image: {bg_image}")
        return np.array(img, dtype=np.float32)

    bg_color = bg_color or "0,255,0"
    parts = bg_color.split(",")
    if len(parts) != 3:
        raise click.BadParameter(
            f"expected 'R,G,B', got {bg_color!r}", param_hint="--bg-color"
        )
    try:
        r, g, b = (int(p) for p in parts)
    except ValueError:
        raise click.BadParameter(
            f"R,G,B values must be integers, got {bg_color!r}", param_hint="--bg-color"
        )
    if not all(0 <= c <= 255 for c in (r, g, b)):
        raise click.BadParameter(
            f"R,G,B values must be in 0-255, got {bg_color!r}", param_hint="--bg-color"
        )
    click.echo(f"[rvm-webcam] Using background color: {r},{g},{b}")
    return np.array([r, g, b], dtype=np.float32)


@click.command()
@click.option("--model-path", type=click.Path(exists=True))
@click.option("--input-device")
@click.option("--output-device")
@click.option("--width", type=int)
@click.option("--height", type=int)
@click.option("--fps", type=int)
@click.option("--downsample-ratio", type=float)
@click.option("--device-id", type=int)
@click.option("--cache-dir", type=click.Path(), default=None, help="MIGraphX .mxr compile cache directory (avoids recompilation on restart)")
@click.option("--bg-color", default=None, help="Composite background as 'R,G,B' (default: 0,255,0). Mutually exclusive with --bg-image.")
@click.option("--bg-image", default=None, type=click.Path(exists=True), help="Composite background image path (JPG/PNG). Mutually exclusive with --bg-color.")
def main(
    model_path,
    input_device,
    output_device,
    width,
    height,
    fps,
    downsample_ratio,
    device_id,
    cache_dir,
    bg_color,
    bg_image,
):
    cfg = _load_config()
    model_path = model_path or cfg.get("model_path") or DEFAULTS["model_path"]
    input_device = input_device or cfg.get("input_device") or DEFAULTS["input_device"]
    output_device = (
        output_device or cfg.get("output_device") or DEFAULTS["output_device"]
    )
    width = width or cfg.get("width") or DEFAULTS["width"]
    height = height or cfg.get("height") or DEFAULTS["height"]
    fps = fps or cfg.get("fps") or DEFAULTS["fps"]
    downsample_ratio = (
        downsample_ratio or cfg.get("downsample_ratio") or DEFAULTS["downsample_ratio"]
    )
    device_id = device_id or cfg.get("device_id") or DEFAULTS["device_id"]
    cache_dir = cache_dir or cfg.get("cache_dir") or DEFAULTS["cache_dir"]

    bg_color_explicit = bg_color or cfg.get("bg_color")
    bg_image = bg_image or cfg.get("bg_image") or DEFAULTS["bg_image"]

    if bg_color_explicit and bg_image:
        raise click.UsageError("--bg-color and --bg-image are mutually exclusive.")

    bg_color = None if bg_image else (bg_color_explicit or DEFAULTS["bg_color"])

    error_q = queue.Queue()

    # 1. Initialize Hardware Engine & Enforce Provider Assertions
    engine = MIGraphXSession(
        model_path,
        height,
        width,
        downsample_ratio=downsample_ratio,
        device_id=device_id,
        cache_dir=cache_dir,
    )
    state_mgr = RecurrentStateBufferGPU(
        height, width, downsample_ratio=downsample_ratio, device_id=device_id
    )

    import pyvirtualcam

    # 2. Instantiate DMA-Capable Pinned Host Output Buffers (FP32)
    # Model has Cast(FP32) nodes on fgr/pha so MIGraphX converts on GPU.
    pb_fgr = allocate_pinned_numpy((1, 3, height, width), dtype=np.float32)
    pb_pha = allocate_pinned_numpy((1, 1, height, width), dtype=np.float32)

    fgr_out = pb_fgr.array
    pha_out = pb_pha.array

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH")

    frame_size = width * height * 3

    vcam = pyvirtualcam.Camera(
        width=width, height=height, fps=fps, device=output_device
    )
    click.echo(
        f"[rvm-webcam] Pipeline successfully bound to virtual camera: {output_device}"
    )

    # 3. Monitor the loopback CAPTURE side so the physical camera (and
    # GPU pipeline) is only activated while a real consumer is
    # streaming from the virtual camera -- i.e. the "shutter" only
    # opens when something is actually watching.
    consumer_monitor = LoopbackConsumerMonitor(output_device)
    if consumer_monitor._supported:
        click.echo(
            "[rvm-webcam] Consumer detection active: physical camera will "
            "only open while a consumer is attached to the virtual camera."
        )

    bg = _load_background(bg_image, bg_color, width, height)
    running = True

    def shutdown(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    frame_count = 0
    t_start = time.perf_counter_ns()

    # Physical-camera capture pipeline state; only non-None while a
    # consumer is attached ("shutter open").
    ffmpeg_proc = None
    grabber = None
    prep = None

    def open_shutter():
        nonlocal ffmpeg_proc, grabber, prep
        subprocess.run(
            [
                "v4l2-ctl",
                "-d",
                input_device,
                "--set-fmt-video",
                f"width={width},height={height},pixelformat=MJPG",
            ],
            capture_output=True,
            timeout=5.0,
        )

        ffmpeg_proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-fflags",
                "nobuffer",
                "-flags",
                "low_delay",
                "-f",
                "v4l2",
                "-input_format",
                "mjpeg",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                input_device,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-an",
                "-sn",
                "-dn",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        proc = ffmpeg_proc

        def read_frame():
            raw = proc.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                return None
            return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))

        grabber = FrameGrabber(read_frame, error_q)
        prep = FramePrep(grabber, width, height, error_q)
        click.echo(f"[rvm-webcam] Shutter open: capturing from {input_device}")

    def close_shutter():
        nonlocal ffmpeg_proc, grabber, prep
        if prep is not None:
            prep.cleanup()
            prep = None
        if grabber is not None:
            grabber.stop()
            grabber = None
        if ffmpeg_proc is not None:
            ffmpeg_proc.terminate()
            try:
                ffmpeg_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                ffmpeg_proc.kill()
                ffmpeg_proc.wait(timeout=2.0)
            ffmpeg_proc = None
            click.echo(f"[rvm-webcam] Shutter closed: released {input_device}")

    # Idle frame sent while no consumer is attached. The virtual
    # camera's OUTPUT side must keep streaming *something* at all
    # times: v4l2loopback's VIDIOC_STREAMON on the CAPTURE side fails
    # with -EIO if the OUTPUT side has never streamed, so gating
    # `vcam.send()` itself behind consumer detection creates a
    # chicken-and-egg deadlock -- the consumer can never connect
    # because the producer waits for the consumer, and the producer
    # never starts because no consumer has connected yet. Only the
    # physical camera capture + GPU inference are gated; the virtual
    # camera output always stays live.
    idle_frame = np.broadcast_to(bg, (height, width, 3)).astype(np.uint8).copy()

    try:
        while running:
            if not error_q.empty():
                raise error_q.get()

            if not consumer_monitor.is_active:
                if prep is not None:
                    close_shutter()
                vcam.send(idle_frame)
                vcam.sleep_until_next_frame()
                continue

            if prep is None:
                open_shutter()

            idx, src_buf = prep.take(timeout=1.0)  # type: ignore[union-attr]
            if src_buf is None:
                continue
            assert idx is not None

            t0 = time.perf_counter_ns()

            try:
                io = engine.io_binding
                io.clear_binding_inputs()
                io.clear_binding_outputs()

                io.bind_cpu_input("src", src_buf)
                state_mgr.bind_inputs(io)

                io.bind_output(
                    "fgr_fp32",
                    "cpu",
                    0,
                    np.float32,
                    fgr_out.shape,
                    fgr_out.ctypes.data,
                )
                io.bind_output(
                    "pha_fp32",
                    "cpu",
                    0,
                    np.float32,
                    pha_out.shape,
                    pha_out.ctypes.data,
                )

                io.bind_output("r1o")
                io.bind_output("r2o")
                io.bind_output("r3o")
                io.bind_output("r4o")

                t1 = time.perf_counter_ns()

                engine.session.run_with_iobinding(io)
                io.synchronize_outputs()

                t2 = time.perf_counter_ns()

                all_out = io.get_outputs()
                state_outputs = [all_out[2], all_out[3], all_out[4], all_out[5]]
                state_mgr.store_outputs(state_outputs)

            finally:
                prep.release(idx)

            pha_sq = pha_out[0, 0, ..., None]
            fgr_sq = fgr_out[0].transpose(1, 2, 0) * 255.0

            com = (fgr_sq * pha_sq + bg * (1.0 - pha_sq)).clip(0, 255).astype(np.uint8)

            vcam.send(com)

            t3 = time.perf_counter_ns()

            frame_dur = (t3 - t0) / 1e9
            target = 1.0 / fps
            if frame_dur < target:
                time.sleep(target - frame_dur)
            t4 = time.perf_counter_ns()

            frame_count += 1

            if frame_count % 100 == 0:
                elapsed_s = (t4 - t_start) / 1e9
                fps_val = frame_count / elapsed_s
                bind_ms = (t1 - t0) / 1e6
                gpu_ms = (t2 - t1) / 1e6
                post_ms = (t3 - t2) / 1e6
                pacing_ms = (t4 - t3) / 1e6

                click.echo(
                    f"[rvm-webcam] {fps_val:.1f} FPS | "
                    f"Bind: {bind_ms:.2f}ms | GPU: {gpu_ms:.2f}ms | "
                    f"Post: {post_ms:.2f}ms | Pacing: {pacing_ms:.2f}ms",
                    err=True,
                )

    finally:
        click.echo("[rvm-webcam] Initiating hardware pipeline teardown...")
        running = False
        close_shutter()
        consumer_monitor.stop()
        vcam.close()

        pb_fgr.cleanup()
        pb_pha.cleanup()

        click.echo(
            "[rvm-webcam] All hardware handles, pinned pages, and streams successfully released."
        )


if __name__ == "__main__":
    main()
