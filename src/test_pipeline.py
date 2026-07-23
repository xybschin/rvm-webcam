"""Isolated unit test suite for rvm-webcam ROCm / MIGraphX pipeline.

Validates safety nets and hardware abstractions without requiring
a physical camera or live GPU.
"""

import queue
import time

import numpy as np
import pytest

from rvm_webcam import (
    FrameGrabber,
    FramePrep,
    LoopbackConsumerMonitor,
    MIGraphXSession,
    PinnedBuffer,
    RecurrentStateBufferGPU,
    WorkerException,
    _load_background,
    _load_hip_runtime,
    allocate_pinned_numpy,
)


# ---------------------------------------------------------------------------
# 1. Memory Allocation & Lifecycle
# ---------------------------------------------------------------------------


class TestPinnedMemory:
    """Verify allocate_pinned_numpy returns a writable PinnedBuffer and
    that cleanup() releases all handles without raising."""

    def test_allocate_returns_writable(self):
        pb = allocate_pinned_numpy((1, 3, 64, 64), dtype=np.float16)
        assert isinstance(pb, PinnedBuffer)
        assert pb.array.shape == (1, 3, 64, 64)
        assert pb.array.dtype == np.float16
        assert pb.array.flags.writeable, "PinnedBuffer array must be writeable"
        # Write into the buffer to confirm it is live memory
        pb.array[:] = 1.0
        assert np.all(pb.array == 1.0)
        pb.cleanup()

    def test_allocate_fallback_no_hip(self, monkeypatch):
        def mock_none():
            return None

        monkeypatch.setattr("rvm_webcam._load_hip_runtime", mock_none)

        pb = allocate_pinned_numpy((2, 4), dtype=np.float32)
        assert pb.is_pinned is False
        assert pb.array.shape == (2, 4)
        assert pb.array.dtype == np.float32
        assert pb.array.flags.writeable
        pb.cleanup()

    def test_cleanup_releases_handles(self):
        pb = allocate_pinned_numpy((1, 16), dtype=np.float32)
        assert pb.is_pinned is True or pb.is_pinned is False  # environment-dependent
        pb.cleanup()
        assert pb.is_pinned is False
        # Calling cleanup a second time must not raise
        pb.cleanup()

    def test_pinnedbuffer_lifecycle_smoke(self):
        pb = allocate_pinned_numpy((4, 4), dtype=np.float32)
        original_addr = pb.addr
        pb.cleanup()
        # After cleanup, is_pinned must be False
        assert pb.is_pinned is False
        # addr should still be accessible (it's an int, not a freed pointer)
        assert pb.addr == original_addr

    def test_cleanup_calls_hip_unregister_and_munlock(self, monkeypatch):
        """Verify PinnedBuffer.cleanup() invokes hipHostUnregister and munlock."""
        from unittest.mock import MagicMock

        mock_hip = MagicMock()
        mock_hip.hipHostRegister.return_value = 0
        mock_hip.hipHostUnregister.return_value = 0

        mock_libc = MagicMock()

        def fake_load_hip():
            return mock_hip

        def fake_cdll(name):
            if name is None:
                return mock_libc
            return MagicMock()

        monkeypatch.setattr("rvm_webcam._load_hip_runtime", fake_load_hip)
        monkeypatch.setattr("ctypes.CDLL", fake_cdll)

        pb = allocate_pinned_numpy((2, 4), dtype=np.float32)
        assert pb.is_pinned is True

        pb.cleanup()

        mock_hip.hipHostUnregister.assert_called_once()
        unregister_arg = mock_hip.hipHostUnregister.call_args[0][0]
        assert unregister_arg.value == pb.addr

        mock_libc.munlock.assert_called_once()
        munlock_args = mock_libc.munlock.call_args[0]
        assert munlock_args[0].value == pb.addr
        assert munlock_args[1].value == pb.bytes_needed


# ---------------------------------------------------------------------------
# 2. Model Signature Validation
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_wrong_type_session():
    """Return a mock InferenceSession whose 'src' input has dtype float32
    instead of the expected float16."""
    import onnxruntime as ort
    from unittest.mock import MagicMock

    fake_session = MagicMock(spec=ort.InferenceSession)
    fake_session.get_providers.return_value = [
        "MIGraphXExecutionProvider",
        "ROCMExecutionProvider",
    ]

    fake_input_src = MagicMock()
    fake_input_src.name = "src"
    fake_input_src.type = "tensor(float32)"  # wrong — pipeline expects float16
    fake_input_src.shape = (1, 3, 720, 1280)

    fake_input_dsr = MagicMock()
    fake_input_dsr.name = "downsample_ratio"
    fake_input_dsr.type = "tensor(float)"
    fake_input_dsr.shape = (1,)

    fake_session.get_inputs.return_value = [fake_input_src, fake_input_dsr]
    fake_session.io_binding.return_value = MagicMock()
    return fake_session


def test_model_validation_dtype_mismatch(monkeypatch, mock_wrong_type_session):
    """MIGraphXSession._validate_model_signature must raise RuntimeError
    when an ONNX input provides float32 instead of float16."""
    import onnxruntime as ort

    monkeypatch.setattr(
        ort, "InferenceSession", lambda *a, **kw: mock_wrong_type_session
    )

    with pytest.raises(RuntimeError, match="type assertion failed.*src"):
        MIGraphXSession("dummy.onnx", 720, 1280, device_id=0)


@pytest.fixture
def mock_wrong_shape_session():
    """Return a mock InferenceSession whose 'src' input has a static
    dimension mismatch (e.g. width=640 instead of 1280)."""
    import onnxruntime as ort
    from unittest.mock import MagicMock

    fake_session = MagicMock(spec=ort.InferenceSession)
    fake_session.get_providers.return_value = [
        "MIGraphXExecutionProvider",
        "ROCMExecutionProvider",
    ]

    fake_input_src = MagicMock()
    fake_input_src.name = "src"
    fake_input_src.type = "tensor(float16)"
    fake_input_src.shape = (1, 3, 720, 640)  # wrong: expected width 1280

    fake_input_dsr = MagicMock()
    fake_input_dsr.name = "downsample_ratio"
    fake_input_dsr.type = "tensor(float)"
    fake_input_dsr.shape = (1,)

    fake_session.get_inputs.return_value = [fake_input_src, fake_input_dsr]
    fake_session.io_binding.return_value = MagicMock()
    return fake_session


def test_model_validation_shape_mismatch(monkeypatch, mock_wrong_shape_session):
    """MIGraphXSession._validate_model_signature must raise RuntimeError
    when an ONNX input has a static dimension mismatch."""
    import onnxruntime as ort

    monkeypatch.setattr(
        ort, "InferenceSession", lambda *a, **kw: mock_wrong_shape_session
    )

    with pytest.raises(RuntimeError, match="shape mismatch.*src"):
        MIGraphXSession("dummy.onnx", 720, 1280, device_id=0)


# ---------------------------------------------------------------------------
# 3. Execution Provider Checks
# ---------------------------------------------------------------------------


def test_recurrent_state_buffer_raises_on_cpu_only(monkeypatch):
    """RecurrentStateBufferGPU must raise RuntimeError when only
    CPUExecutionProvider is available."""
    import onnxruntime as ort

    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )

    with pytest.raises(RuntimeError, match="Execution provider assertion failed"):
        RecurrentStateBufferGPU(720, 1280, device_id=0)


def test_migraphx_session_cpu_only_raises(monkeypatch):
    """MIGraphXSession must raise RuntimeError when no GPU execution
    provider is available."""
    import onnxruntime as ort

    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )

    with pytest.raises(RuntimeError, match="MIGraphX execution provider not available"):
        MIGraphXSession("dummy.onnx", 720, 1280, device_id=0)


# ---------------------------------------------------------------------------
# 4. Worker Thread Exception Propagation
# ---------------------------------------------------------------------------


def failing_read_fn():
    raise ValueError("camera read failure simulated")


def test_error_queue_captures_frame_grabber_exception():
    """FrameGrabber background thread must catch exceptions and push them
    into error_q for the main loop to re-raise."""
    error_q: queue.Queue = queue.Queue()

    grabber = FrameGrabber(failing_read_fn, error_q)
    time.sleep(0.3)
    grabber.stop()

    assert not error_q.empty(), "error_q should contain the worker exception"
    exc = error_q.get()
    assert isinstance(exc, WorkerException)
    assert "FrameGrabber failed" in str(exc)
    assert "camera read failure simulated" in str(exc)


def test_main_loop_raises_worker_exception():
    """Simulate the main loop pattern: check error_q and raise."""
    error_q: queue.Queue = queue.Queue()
    error_q.put(WorkerException("FramePrep failed: camera read failure simulated"))

    with pytest.raises(WorkerException, match="FramePrep failed"):
        exc = error_q.get()
        raise exc


def test_frame_prep_exception_propagation():
    """FramePrep background thread must catch exceptions and push them
    into error_q."""
    error_q: queue.Queue = queue.Queue()
    grabber = FrameGrabber(failing_read_fn, error_q)

    # FramePrep will internally try to grab frames from the failing grabber.
    # The grabber's exception hits error_q, and FramePrep's own thread
    # should also handle its own exceptions.
    prep = FramePrep(grabber, 64, 64, error_q)
    time.sleep(0.5)
    prep.cleanup()
    grabber.stop()

    # At least one WorkerException should be in the queue
    found = False
    while not error_q.empty():
        exc = error_q.get()
        if isinstance(exc, WorkerException):
            found = True
            break
    assert found, "Expected at least one WorkerException in error_q"


# ---------------------------------------------------------------------------
# 5. HIP Runtime Loader Fallback (bonus coverage)
# ---------------------------------------------------------------------------


def test_load_hip_runtime_fallback_returns_none(monkeypatch):
    """When libamdhip64.so is not reachable, _load_hip_runtime returns None
    rather than raising."""
    import ctypes

    # Make all resolution strategies fail
    monkeypatch.setattr(
        ctypes, "CDLL", lambda _: (_ for _ in ()).throw(OSError("not found"))
    )
    monkeypatch.setattr("ctypes.util.find_library", lambda _: None)
    monkeypatch.setattr("os.path.isfile", lambda _: False)

    result = _load_hip_runtime()
    assert result is None


# ---------------------------------------------------------------------------
# 6. Loopback Consumer Detection ("shutter only opens for a real consumer")
# ---------------------------------------------------------------------------


def test_loopback_consumer_monitor_fails_open_on_bad_device():
    """If the device can't be opened/subscribed to, the monitor must
    fail open (is_active == True) so the pipeline never silently stops
    producing frames on unsupported platforms/devices."""
    mon = LoopbackConsumerMonitor("/dev/this-device-does-not-exist-xyz")
    assert mon._supported is False
    assert mon.is_active is True
    mon.stop()


@pytest.fixture
def loopback_device():
    """Locate a v4l2loopback CAPTURE device to test consumer-detection
    against. Skips the test if none is available (e.g. CI without the
    kernel module loaded)."""
    import glob
    import os as _os

    for path in sorted(glob.glob("/sys/class/video4linux/video*")):
        name_path = _os.path.join(path, "name")
        try:
            with open(name_path) as f:
                pass
        except OSError:
            continue
        dev = "/dev/" + _os.path.basename(path)
        try:
            fd = _os.open(dev, _os.O_RDWR | _os.O_NONBLOCK)
            _os.close(fd)
        except OSError:
            continue
        # crude v4l2loopback detection: sysfs 'state' attr only exists there
        if _os.path.exists(_os.path.join(path, "state")):
            return dev
    pytest.skip("no accessible v4l2loopback device found for live test")


def test_loopback_consumer_monitor_detects_real_consumer(loopback_device):
    """End-to-end: subscribing via LoopbackConsumerMonitor must report
    is_active=False with no consumer, flip to True while a real V4L2
    CAPTURE consumer (ffmpeg) is streaming, and flip back to False
    once that consumer disconnects."""
    import shutil
    import subprocess
    import threading

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available for live consumer test")

    try:
        import pyvirtualcam
    except ImportError:
        pytest.skip("pyvirtualcam not available for live producer test")

    mon = LoopbackConsumerMonitor(loopback_device)
    if not mon._supported:
        pytest.skip(
            "V4L2_EVENT_PRI_CLIENT_USAGE not supported on this device/kernel"
        )

    try:
        assert mon.is_active is False

        cam = pyvirtualcam.Camera(
            width=640, height=480, fps=15, device=loopback_device
        )
        stop_producer = threading.Event()
        producer_thread = None
        try:
            frame = np.zeros((480, 640, 3), np.uint8)

            def produce():
                while not stop_producer.is_set():
                    try:
                        cam.send(frame)
                        cam.sleep_until_next_frame()
                    except Exception:
                        break

            producer_thread = threading.Thread(target=produce, daemon=True)
            producer_thread.start()
            time.sleep(1.0)

            assert mon.is_active is False, "no consumer yet, should be inactive"

            proc = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "v4l2",
                    "-i",
                    loopback_device,
                    "-f",
                    "null",
                    "-",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            try:
                time.sleep(1.5)
                if proc.poll() is not None:
                    stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                    pytest.skip(
                        f"ffmpeg consumer could not attach to {loopback_device} "
                        f"(likely already in use by another application): {stderr.strip()}"
                    )
                assert mon.is_active is True, "consumer attached, should be active"
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()

            time.sleep(1.5)
            assert mon.is_active is False, "consumer gone, should be inactive again"
        finally:
            stop_producer.set()
            if producer_thread is not None:
                producer_thread.join(timeout=3)
            cam.close()
    finally:
        mon.stop()


# ---------------------------------------------------------------------------
# 7. Composite Background Loading (--bg-color / --bg-image)
# ---------------------------------------------------------------------------


def test_load_background_default_color():
    """_load_background with no image falls back to the given bg_color
    string, returning a broadcastable (3,) RGB array."""
    bg = _load_background(None, "0,255,0", width=100, height=50)
    assert bg.shape == (3,)
    assert bg.dtype == np.float32
    assert np.array_equal(bg, [0, 255, 0])


def test_load_background_custom_color():
    bg = _load_background(None, "10,20,30", width=100, height=50)
    assert np.array_equal(bg, [10, 20, 30])


def test_load_background_color_wrong_arity():
    import click

    with pytest.raises(click.BadParameter, match="expected 'R,G,B'"):
        _load_background(None, "10,20", width=100, height=50)


def test_load_background_color_non_integer():
    import click

    with pytest.raises(click.BadParameter, match="must be integers"):
        _load_background(None, "a,b,c", width=100, height=50)


def test_load_background_color_out_of_range():
    import click

    with pytest.raises(click.BadParameter, match="must be in 0-255"):
        _load_background(None, "300,20,30", width=100, height=50)


def test_load_background_image(tmp_path):
    """_load_background with bg_image loads, converts to RGB, and resizes
    to the target (height, width) -- returning a full-frame array rather
    than a broadcastable color."""
    from PIL import Image

    img_path = tmp_path / "bg.png"
    Image.new("RGB", (200, 100), color=(50, 100, 150)).save(img_path)

    bg = _load_background(str(img_path), None, width=1280, height=720)
    assert bg.shape == (720, 1280, 3)
    assert bg.dtype == np.float32
    assert np.allclose(bg[0, 0], [50, 100, 150])
    assert np.allclose(bg[-1, -1], [50, 100, 150])


def test_load_background_image_missing_file():
    import click

    with pytest.raises(click.BadParameter, match="Could not load image"):
        _load_background("/nonexistent/path/to/image.png", None, width=100, height=50)


def test_composite_math_with_image_background(tmp_path):
    """Sanity-check the fgr*pha + bg*(1-pha) formula the main loop uses,
    confirming a full-frame image background broadcasts correctly
    against a (H, W, 1) alpha channel."""
    from PIL import Image

    img_path = tmp_path / "bg.png"
    Image.new("RGB", (10, 10), color=(50, 100, 150)).save(img_path)
    bg = _load_background(str(img_path), None, width=10, height=10)

    pha_sq = np.full((10, 10, 1), 0.5, dtype=np.float32)
    fgr_sq = np.full((10, 10, 3), 200.0, dtype=np.float32)
    com = (fgr_sq * pha_sq + bg * (1.0 - pha_sq)).clip(0, 255).astype(np.uint8)

    assert com.shape == (10, 10, 3)
    expected_r = int(200 * 0.5 + 50 * 0.5)
    assert abs(int(com[0, 0, 0]) - expected_r) <= 1
