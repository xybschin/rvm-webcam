#!/usr/bin/env python3
"""rvm-webcam: real-time background removal virtual camera using RobustVideoMatting."""

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import click
import cv2
import numpy as np
import torch


def load_model(model_path: str, backbone: str, device: str):
    model = torch.hub.load("PeterL1n/RobustVideoMatting", backbone).eval().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def open_capture(device: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open capture device {device}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # cuts USB bandwidth at high res
    # Minimize kernel-side buffering so the grabber always reads the freshest frame.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_w, actual_h) != (width, height):
        click.echo(
            f"[rvm-webcam] WARNING: requested {width}x{height} but device reports "
            f"{actual_w}x{actual_h}; frames will be resized.",
            err=True,
        )
    return cap


class FrameGrabber:
    """Background thread that drains the V4L2 queue and retains only the freshest frame.

    When inference can't keep pace with the camera, OpenCV's ``cap.read()`` returns
    frames that have been sitting in kernel buffers, so the pipeline processes stale
    data and effective throughput collapses. This thread continuously reads from the
    capture device and overwrites a single slot, so the consumer always gets the most
    recent frame and stale frames are dropped automatically.
    """

    def __init__(self, cap):
        self.cap = cap
        self._cond = threading.Condition()
        self._frame = None
        self._new = False
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            if not ret:
                # Brief yield on read failure to avoid a busy spin.
                time.sleep(0.001)
                continue
            with self._cond:
                self._frame = frame
                self._new = True
                self._cond.notify_all()

    def wait_new(self, timeout=1.0):
        """Block until a frame newer than the last consumed one is available, then
        return it (or None on timeout). Implicitly rate-limits the consumer to the
        camera's framerate when the consumer is faster than the camera."""
        with self._cond:
            if not self._new:
                self._cond.wait(timeout=timeout)
                if not self._new:
                    return None
            self._new = False
            return self._frame

    def stop(self):
        self._stopped = True
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)


class FramePrep:
    """Pipelined CPU preparation (resize + BGR->RGB + NCHW pack) on a background thread.

    Runs concurrently with GPU inference: while the main thread is running the model
    for frame N, this thread is preparing frame N+1. Prepared frames land in one of two
    pinned uint8 NCHW buffers (double-buffered) so H2D transfers can be issued with
    ``non_blocking=True`` and overlap with the previous frame's D2H + vcam.send.
    Only the most recently prepared buffer is kept; older unconsumed ones are dropped.
    """

    def __init__(self, grabber: FrameGrabber, width: int, height: int):
        self.grabber = grabber
        self.W = width
        self.H = height
        self.bufs = [
            torch.empty((1, 3, height, width), dtype=torch.uint8, pin_memory=True),
            torch.empty((1, 3, height, width), dtype=torch.uint8, pin_memory=True),
        ]
        self._cond = threading.Condition()
        self._ready_idx = None  # index of buffer holding the freshest prepared frame
        self._inuse_idx = None  # index currently held by the main thread for H2D
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _free_idx(self):
        for i in (0, 1):
            if i != self._ready_idx and i != self._inuse_idx:
                return i
        return None

    def _run(self):
        while not self._stopped:
            frame = self.grabber.wait_new(timeout=1.0)
            if frame is None:
                continue
            if frame.shape[1] != self.W or frame.shape[0] != self.H:
                frame = cv2.resize(frame, (self.W, self.H))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Permute to NCHW and make contiguous; this is the only per-frame CPU alloc.
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
            with self._cond:
                fi = None
                while not self._stopped:
                    fi = self._free_idx()
                    if fi is not None:
                        break
                    self._cond.wait(timeout=1.0)
                if fi is None:
                    return
                self.bufs[fi].copy_(t)
                self._ready_idx = fi  # overwrites any older unconsumed ready buffer
                self._cond.notify_all()

    def take(self, timeout=5.0):
        """Block until a prepared frame is available; returns (index, buffer).
        Marks the buffer in-use so the prep thread won't reuse it, and clears ready
        so the next prepared frame is always the freshest."""
        with self._cond:
            while self._ready_idx is None and not self._stopped:
                if not self._cond.wait(timeout=timeout):
                    if self._ready_idx is None:
                        continue
            if self._stopped or self._ready_idx is None:
                return None, None
            idx = self._ready_idx
            self._ready_idx = None
            self._inuse_idx = idx
            return idx, self.bufs[idx]

    def release(self, idx):
        """Mark the in-use buffer free again once its H2D transfer has completed."""
        with self._cond:
            if self._inuse_idx == idx:
                self._inuse_idx = None
            self._cond.notify_all()

    def stop(self):
        self._stopped = True
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)


class CapturePipeline:
    """Owns the capture device plus its grabber/prep helper threads as a unit."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.cap = None
        self.grabber = None
        self.prep = None

    def start(self, input_device: str, fps: int):
        self.cap = open_capture(input_device, self.width, self.height, fps)
        self.grabber = FrameGrabber(self.cap)
        self.prep = FramePrep(self.grabber, self.width, self.height)

    def stop(self):
        if self.prep is not None:
            self.prep.stop()
        if self.grabber is not None:
            self.grabber.stop()
        if self.cap is not None:
            self.cap.release()
        self.prep = self.grabber = self.cap = None


def _consumer_count(output_device: str) -> int:
    our_pid = os.getpid()
    count = 0
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == our_pid:
                continue
            try:
                fd_dir = f"/proc/{entry}/fd"
                for fd_entry in os.listdir(fd_dir):
                    try:
                        link = os.readlink(f"{fd_dir}/{fd_entry}")
                        if link == output_device:
                            count += 1
                    except OSError:
                        pass
            except (PermissionError, FileNotFoundError):
                pass
    except PermissionError:
        pass
    return count


DEFAULTS = {
    "backbone": "mobilenetv3",
    "input_device": "/dev/video0",
    "output_device": "/dev/video10",
    "width": 1280,
    "height": 720,
    "fps": 30,
    "downsample_ratio": 0.25,
    "precision": "auto",
}


@click.command()
@click.option("--model-path", required=False, type=click.Path(exists=True), help="Path to rvm_mobilenetv3.pth (can also be set in config.json)")
@click.option("--backbone", default=None, type=click.Choice(["mobilenetv3", "resnet50"]))
@click.option("--input-device", default=None, help="Real webcam device")
@click.option("--output-device", default=None, help="v4l2loopback virtual device")
@click.option("--width", default=None, type=int)
@click.option("--height", default=None, type=int)
@click.option("--fps", default=None, type=int)
@click.option("--downsample-ratio", default=None, type=float, help="Lower = faster, less edge detail")
@click.option("--bg-color", default=None, help="R,G,B for composite background (mutually exclusive with --bg-image)")
@click.option("--bg-image", default=None, type=click.Path(exists=True), help="Path to a background image, e.g. JPG/PNG (mutually exclusive with --bg-color)")
@click.option("--precision", default=None, type=click.Choice(["auto", "fp16", "fp32"]))
@click.option("--compile", "use_compile", is_flag=True, default=False, help="Apply torch.compile (requires PyTorch ≥ 2.0, may improve GPU performance)")
@click.option("--preview", is_flag=True, default=False, help="Write raw RGB24 frames to stdout for piping to ffplay (e.g. | ffplay ...). All logging goes to stderr.")
@click.option("--on-demand", is_flag=True, default=False, help="Only capture webcam when a consumer reads /dev/video10")
def main(model_path, backbone, input_device, output_device, width, height, fps,
         downsample_ratio, bg_color, bg_image, precision, use_compile, preview, on_demand):
    cfg = {}
    for config_path in [
        Path.home() / ".config" / "rvm-webcam" / "config.json",
        Path("/etc/rvm-webcam/config.json"),
    ]:
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            break

    model_path = model_path or cfg.get("model_path")
    if model_path is None:
        raise click.UsageError(
            "--model-path is required. Pass it directly or create "
            "~/.config/rvm-webcam/config.json or /etc/rvm-webcam/config.json "
            'with {"model_path": "/path/to/model.pth"}'
        )

    backbone = backbone or cfg.get("backbone") or DEFAULTS["backbone"]
    input_device = input_device or cfg.get("input_device") or DEFAULTS["input_device"]
    output_device = output_device or cfg.get("output_device") or DEFAULTS["output_device"]
    width = width or cfg.get("width") or DEFAULTS["width"]
    height = height or cfg.get("height") or DEFAULTS["height"]
    fps = fps or cfg.get("fps") or DEFAULTS["fps"]
    downsample_ratio = downsample_ratio or cfg.get("downsample_ratio") or DEFAULTS["downsample_ratio"]
    bg_color = bg_color or cfg.get("bg_color")
    bg_image = bg_image or cfg.get("bg_image")
    precision = precision or cfg.get("precision") or DEFAULTS["precision"]
    if not use_compile:
        use_compile = cfg.get("compile", False)
    if not preview:
        preview = cfg.get("preview", False)
    if not on_demand:
        on_demand = cfg.get("on_demand", False)

    if bg_color and bg_image:
        raise click.UsageError("--bg-color and --bg-image are mutually exclusive.")

    _real_stdout = None
    if preview:
        if sys.stdout.isatty():
            click.echo(
                "[rvm-webcam] ERROR: --preview requires piping stdout to ffplay, e.g.:\n"
                f"  {sys.argv[0]} --model-path ... --preview | ffplay -f rawvideo "
                f"-pixel_format rgb24 -video_size {width}x{height} -i -",
                err=True,
            )
            raise SystemExit(1)
        _real_stdout = sys.stdout
        sys.stdout = sys.stderr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _is_rocm = (
        torch.cuda.is_available()
        and hasattr(torch.version, "hip")
        and torch.version.hip is not None
    )
    if precision == "auto":
        dtype = torch.float16 if device == "cuda" else torch.float32
    else:
        dtype = torch.float16 if precision == "fp16" else torch.float32

    _backend = "rocm" if _is_rocm else device
    click.echo(f"[rvm-webcam] device={_backend} dtype={dtype} downsample_ratio={downsample_ratio}")

    # Fixed input shape -> let cuDNN/MIOpen pick the fastest kernels.
    if not _is_rocm:
        torch.backends.cudnn.benchmark = True

    model = load_model(model_path, backbone, device)
    # Convert model weights to compute dtype so they match src_gpu without autocast.
    model = model.to(dtype)
    original_model = model  # kept for torch.compile runtime fallback

    if use_compile:
        if not hasattr(torch, "compile"):
            click.echo("[rvm-webcam] WARNING: torch.compile not available (requires PyTorch >= 2.0), skipping.", err=True)
            use_compile = False
        else:
            original_model = model
            model = torch.compile(model, mode="reduce-overhead")
            click.echo("[rvm-webcam] torch.compile enabled")

    if bg_image:
        img = cv2.imread(bg_image)
        if img is None:
            raise click.BadParameter(f"Could not load image: {bg_image}", param_hint="--bg-image")
        img = cv2.resize(img, (width, height))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bg_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(device=device, dtype=dtype) / 255.0
        click.echo(f"[rvm-webcam] using background image: {bg_image}")
    else:
        color_str = bg_color if bg_color else "0,255,0"
        parts = color_str.split(",")
        if len(parts) != 3:
            raise click.BadParameter(
                f"expected 'R,G,B', got {color_str!r}", param_hint="--bg-color"
            )
        try:
            r, g, b = (int(x) for x in parts)
        except ValueError:
            raise click.BadParameter(
                f"R,G,B values must be integers, got {color_str!r}", param_hint="--bg-color"
            )
        if not all(0 <= c <= 255 for c in (r, g, b)):
            raise click.BadParameter(
                f"R,G,B values must be in 0-255, got {color_str!r}", param_hint="--bg-color"
            )
        bg_tensor = torch.tensor([r / 255, g / 255, b / 255], device=device, dtype=dtype).view(3, 1, 1)
        click.echo(f"[rvm-webcam] using background color: {color_str}")

    import pyvirtualcam
    vcam = pyvirtualcam.Camera(width=width, height=height, fps=fps, device=output_device)
    click.echo(f"[rvm-webcam] virtual camera live on {output_device}")

    use_cuda = device == "cuda"

    # Two ping-pong GPU src tensors so H2D of frame N+1 can overlap with the
    # inference / composite / D2H of frame N on the default stream.
    src_gpu = [torch.empty((1, 3, height, width), device=device, dtype=dtype) for _ in range(2)]
    h2d_stream = torch.cuda.Stream() if use_cuda else None
    h2d_events: list[torch.cuda.Event] = (
        [torch.cuda.Event(), torch.cuda.Event()] if use_cuda else []
    )

    def launch_h2d(buf, dst, slot):
        if use_cuda:
            with torch.cuda.stream(h2d_stream):
                dst.copy_(buf, non_blocking=True)
                dst.div_(255.0)
            h2d_events[slot].record(h2d_stream)
        else:
            dst.copy_(buf)
            dst.div_(255.0)

    pipeline = CapturePipeline(width, height)

    # Defined unconditionally so the on-demand branch below is always well-formed;
    # they are only read while running in on-demand mode.
    last_poll = 0.0
    POLL_INTERVAL = 1.0
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    if on_demand:
        click.echo(
            f"[rvm-webcam] on-demand enabled: webcam opens when consumer reads {output_device}"
        )
    else:
        pipeline.start(input_device, fps)

    rec = [None] * 4
    running = True

    def shutdown(signum, frame):
        nonlocal running
        click.echo("\n[rvm-webcam] shutting down...")
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    frame_count = 0
    t_start = time.time()
    # Stage timing accumulators (in nanoseconds).
    acc_h2d = 0
    acc_infer = 0
    acc_take = 0
    acc_d2h = 0
    acc_sleep = 0
    # Two-slot ping-pong: src_gpu[cur] holds the frame being inferred this
    # iteration; its H2D was launched at the tail of the previous iteration.
    # prev_buf_idx is the prep buffer still held (in-use) for that H2D.
    cur = 0
    prev_buf_idx = None

    try:
        with torch.no_grad():
            while running:
                if on_demand:
                    now = time.monotonic()
                    if now - last_poll >= POLL_INTERVAL:
                        last_poll = now
                        consumers = _consumer_count(output_device)

                        if consumers > 0:
                            if pipeline.cap is None:
                                click.echo(
                                    f"[rvm-webcam] {consumers} consumer(s) connected, opening webcam",
                                    err=True,
                                )
                                try:
                                    pipeline.start(input_device, fps)
                                except RuntimeError as e:
                                    click.echo(f"[rvm-webcam] {e}, retrying...", err=True)
                                    pipeline.stop()
                                    continue
                                rec = [None] * 4
                                frame_count = 0
                                t_start = time.time()
                                acc_h2d = acc_infer = acc_take = acc_d2h = acc_sleep = 0
                                cur = 0
                                prev_buf_idx = None
                        else:
                            if pipeline.cap is not None:
                                click.echo(
                                    "[rvm-webcam] no consumers, releasing webcam",
                                    err=True,
                                )
                                pipeline.stop()
                                rec = [None] * 4

                    if pipeline.cap is None:
                        vcam.send(black_frame)
                        vcam.sleep_until_next_frame()
                        continue

                # cap is now known to be open -> prep and grabber are live.
                prep = pipeline.prep
                assert prep is not None

                if prev_buf_idx is None:
                    # Bootstrap: pull the first prepared frame and launch its H2D.
                    # The next iteration will infer it.
                    idx, buf = prep.take(timeout=5.0)
                    if buf is None:
                        continue
                    launch_h2d(buf, src_gpu[cur], cur)
                    prev_buf_idx = idx
                    continue

                # 1. Ensure the current frame's H2D has landed on src_gpu[cur], then
                #    release its source buffer so the prep thread can refill it.
                t0 = time.perf_counter_ns()
                if use_cuda:
                    h2d_events[cur].wait()
                prep.release(prev_buf_idx)
                t1 = time.perf_counter_ns()
                acc_h2d += t1 - t0

                # 2. Inference + composite for the current frame. While this runs on
                #    the default stream, the prep thread prepares frame N+1 on the CPU.
                try:
                    fgr, pha, *rec = model(src_gpu[cur], *rec, downsample_ratio)
                except Exception:
                    if use_compile:
                        click.echo(
                            "[rvm-webcam] WARNING: torch.compile failed at runtime; "
                            "falling back to uncompiled model",
                            err=True,
                        )
                        model = original_model
                        use_compile = False
                        fgr, pha, *rec = model(src_gpu[cur], *rec, downsample_ratio)
                    else:
                        raise
                t2 = time.perf_counter_ns()
                acc_infer += t2 - t1

                com = fgr * pha + bg_tensor * (1 - pha)
                out_gpu = (com[0].permute(1, 2, 0).clamp(0, 1) * 255).byte()

                # 3. Take the next prepared buffer and launch its H2D on the side
                #    stream. This transfer overlaps with the D2H + vcam.send below.
                nxt = 1 - cur
                idx, buf = prep.take(timeout=5.0)
                t3 = time.perf_counter_ns()
                acc_take += t3 - t2
                if buf is None:
                    # Shutting down; still emit the current frame before bailing.
                    out = out_gpu.cpu().numpy()
                    vcam.send(out)
                    if _real_stdout is not None:
                        _real_stdout.buffer.write(out.tobytes())
                        _real_stdout.buffer.flush()
                    break
                launch_h2d(buf, src_gpu[nxt], nxt)

                # 4. D2H + send (default stream) runs concurrently with the H2D of
                #    frame N+1 already issued on the side stream in step 3.
                out = out_gpu.cpu().numpy()
                vcam.send(out)

                if _real_stdout is not None:
                    _real_stdout.buffer.write(out.tobytes())
                    _real_stdout.buffer.flush()

                frame_count += 1
                prev_buf_idx = idx
                cur = nxt

                t4 = time.perf_counter_ns()
                vcam.sleep_until_next_frame()
                t5 = time.perf_counter_ns()
                acc_sleep += t5 - t4
                acc_d2h += t4 - t3

                if frame_count % 100 == 0 and frame_count > 0:
                    n = frame_count
                    ms = lambda ns: ns / 1e6
                    click.echo(
                        f"[rvm-webcam] {frame_count / (t5 / 1e9):.1f} fps  "
                        f"h2d={ms(acc_h2d/n):.1f}  "
                        f"infer={ms(acc_infer/n):.1f}  "
                        f"take={ms(acc_take/n):.1f}  "
                        f"d2h+send={ms(acc_d2h/n):.1f}  "
                        f"sleep={ms(acc_sleep/n):.1f}",
                        err=True,
                    )
    finally:
        pipeline.stop()
        vcam.close()
        click.echo("[rvm-webcam] released camera and virtual device, exiting.")


if __name__ == "__main__":
    main()
