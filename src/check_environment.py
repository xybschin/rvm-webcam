#!/usr/bin/env python3
"""cli verification tool: check system dependencies for rvm-webcam"""

import glob
import os
import shutil
import subprocess
import sys


def _ok(msg: str, indent: int = 2):
    _print("✓", msg, "32", indent)


def _warn(msg: str, indent: int = 2):
    _print("!", msg, "33", indent)


def _fail(msg: str, indent: int = 2):
    _print("✗", msg, "31", indent)


def _print(symbol: str, msg: str, color: str, indent: int):
    sys.stderr.write(f"\033[{color}m{' ' * indent}{symbol} {msg}\033[0m\n")


def check_video_devices() -> bool:
    ok = True
    devices = sorted(glob.glob("/dev/video*"))
    if not devices:
        _fail("No /dev/video* devices found")
        return False

    _ok(f"Found {len(devices)} video device(s):")
    for d in devices[:8]:
        try:
            st = os.stat(d)
            mode = st.st_mode & 0o777
            perms = oct(mode)
            owner = f"uid={st.st_uid}"
            readable = bool(mode & 0o444)
            _ok(f"  {d}  permissions={perms}  {owner}  readable={readable}")
            if not readable:
                _warn(f"  {d} is not world-readable (may need udev rules)")
                ok = False
        except OSError as e:
            _fail(f"  Cannot stat {d}: {e}")
            ok = False

    if len(devices) > 8:
        _ok(f"  ... and {len(devices) - 8} more")
    return ok


def check_v4l2loopback() -> bool:
    try:
        result = subprocess.run(["lsmod"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if "v4l2loopback" in line:
                _ok(f"v4l2loopback kernel module loaded ({line.strip()})")
                return True
        _fail("v4l2loopback kernel module NOT loaded")
        _warn("Run: sudo modprobe v4l2loopback")
        return False
    except FileNotFoundError:
        _fail("lsmod not found (not a Linux system?)")
        return False
    except subprocess.TimeoutExpired:
        _fail("lsmod timed out")
        return False


def check_rocm_driver() -> bool:
    ok = True

    # Check /dev/kfd (ROCk driver) and /dev/dri/render* (drm)
    kfd_exists = os.path.exists("/dev/kfd")
    render_nodes = glob.glob("/dev/dri/renderD*")
    if kfd_exists:
        _ok("/dev/kfd present (ROCk kernel driver)")
    else:
        _fail("/dev/kfd NOT found — ROCm kernel driver may not be loaded")

    if render_nodes:
        _ok(f"DRM render nodes present: {len(render_nodes)} found")
    else:
        _fail("No /dev/dri/renderD* nodes — GPU driver may not be loaded")
        ok = False

    # Check rocm-smi
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi:
        try:
            result = subprocess.run(
                [rocm_smi, "--showproductname"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.strip():
                        _ok(f"  rocm-smi: {line.strip()}")
            else:
                _warn(f"rocm-smi returned {result.returncode}")
        except subprocess.TimeoutExpired:
            _warn("rocm-smi timed out")
    else:
        _warn("rocm-smi not found in PATH")

    # Check libamdhip64 (lazy import to avoid pulling in heavy deps)
    import importlib

    hip_loader = importlib.import_module("rvm_webcam")._load_hip_runtime
    hip = hip_loader()
    if hip is not None:
        _ok("libamdhip64.so resolved")
    else:
        _fail("libamdhip64.so could not be loaded")
        ok = False

    return ok if (kfd_exists or render_nodes) else False


def check_ffmpeg() -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            result = subprocess.run(
                [ffmpeg, "-version"], capture_output=True, text=True, timeout=5
            )
            first = result.stdout.splitlines()[0] if result.stdout else "unknown"
            _ok(f"ffmpeg found: {first}")
            return True
        except (subprocess.TimeoutExpired, OSError):
            _fail("ffmpeg found but failed to execute")
            return False
    else:
        _fail("ffmpeg not found in PATH")
        return False


def check_ort_providers() -> bool:
    import onnxruntime as ort

    providers = ort.get_available_providers()
    _ok(f"ONNX Runtime version: {ort.__version__}")
    _ok(f"Available providers ({len(providers)}):")

    has_gpu = False
    gpu_keywords = ["migraphx", "rocm", "cuda", "tensorrt"]
    for p in providers:
        _ok(f"  {p}")
        if any(kw in p.lower() for kw in gpu_keywords):
            has_gpu = True

    if has_gpu:
        _ok("GPU execution provider detected")
    else:
        _fail(
            "No GPU execution provider found — only CPU is available. "
            "Install onnxruntime-rocm or onnxruntime-gpu."
        )

    return has_gpu


def main() -> int:
    sys.stderr.write("═══ rvm-webcam Environment Check ═══\n\n")
    all_ok = True

    sys.stderr.write("── Video Devices ──\n")
    all_ok &= check_video_devices()
    sys.stderr.write("\n")

    sys.stderr.write("── v4l2loopback ──\n")
    all_ok &= check_v4l2loopback()
    sys.stderr.write("\n")

    sys.stderr.write("── ROCm / GPU Driver ──\n")
    all_ok &= check_rocm_driver()
    sys.stderr.write("\n")

    sys.stderr.write("── ffmpeg ──\n")
    all_ok &= check_ffmpeg()
    sys.stderr.write("\n")

    sys.stderr.write("── ONNX Runtime Providers ──\n")
    all_ok &= check_ort_providers()
    sys.stderr.write("\n")

    if all_ok:
        _ok("All environment checks passed.", indent=0)
        return 0
    else:
        _fail("Some checks failed — review warnings above.", indent=0)
        return 1


if __name__ == "__main__":
    sys.exit(main())
