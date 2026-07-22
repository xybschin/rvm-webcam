#!/usr/bin/env python3
"""rvm-webcam: real-time background removal virtual camera using RobustVideoMatting."""

import json
import os
import signal
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
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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
                time.sleep(0.001)
                continue
            with self._cond:
                self._frame = frame
                self._new = True
                self._cond.notify_all()

    def wait_new(self, timeout=1.0):
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
    def __init__(self, grabber: FrameGrabber, width: int, height: int):
        self.grabber = grabber
        self.W = width
        self.H = height
        self.bufs = [
            torch.empty((1, 3, height, width), dtype=torch.uint8, pin_memory=True),
            torch.empty((1, 3, height, width), dtype=torch.uint8, pin_memory=True),
        ]
        self._cond = threading.Condition()
        self._ready_idx = None
        self._inuse_idx = None
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
                self._ready_idx = fi
                self._cond.notify_all()

    def take(self, timeout=5.0):
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


def _draw_debug_overlay(img, fps, infer_ms, take_ms, d2h_ms):
    text = f"{fps:.1f} fps  I:{infer_ms:.1f}  D:{d2h_ms:.1f}  C:{take_ms:.1f}ms"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    pad = 8
    margin = 10
    h, w = img.shape[:2]
    x1 = margin
    y1 = h - margin - th - bl - 2 * pad
    x2 = x1 + tw + 2 * pad
    y2 = h - margin
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    tx = x1 + pad
    ty = y2 - pad - bl
    cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _load_config():
    for config_path in [
        Path.home() / ".config" / "rvm-webcam" / "config.json",
        Path("/etc/rvm-webcam/config.json"),
    ]:
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
    return {}


def _compute_bg_tensor(bg_image, bg_color, width, height, device, dtype):
    if bg_image:
        img = cv2.imread(bg_image)
        if img is None:
            raise click.BadParameter(f"Could not load image: {bg_image}", param_hint="--bg-image")
        img = cv2.resize(img, (width, height))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bg_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(device=device, dtype=dtype) / 255.0
        click.echo(f"[rvm-webcam] using background image: {bg_image}")
        return bg_tensor

    color_str = bg_color if bg_color else "0,255,0"
    parts = color_str.split(",")
    if len(parts) != 3:
        raise click.BadParameter(f"expected 'R,G,B', got {color_str!r}", param_hint="--bg-color")
    try:
        r, g, b = (int(x) for x in parts)
    except ValueError:
        raise click.BadParameter(f"R,G,B values must be integers, got {color_str!r}", param_hint="--bg-color")
    if not all(0 <= c <= 255 for c in (r, g, b)):
        raise click.BadParameter(f"R,G,B values must be in 0-255, got {color_str!r}", param_hint="--bg-color")
    click.echo(f"[rvm-webcam] using background color: {color_str}")
    return torch.tensor([r / 255, g / 255, b / 255], device=device, dtype=dtype).view(3, 1, 1)


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
@click.option("--on-demand", is_flag=True, default=False, help="Only capture webcam when a consumer reads /dev/video10")
@click.option("--debug", is_flag=True, default=False, help="Overlay performance stats on the output frame")
def main(model_path, backbone, input_device, output_device, width, height, fps,
         downsample_ratio, bg_color, bg_image, precision, on_demand, debug):
    cfg = _load_config()

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
    if not on_demand:
        on_demand = cfg.get("on_demand", False)
    if not debug:
        debug = cfg.get("debug", False)

    if bg_color and bg_image:
        raise click.UsageError("--bg-color and --bg-image are mutually exclusive.")

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

    if not _is_rocm:
        torch.backends.cudnn.benchmark = True

    model = load_model(model_path, backbone, device)
    model = model.to(dtype)

    bg_tensor = _compute_bg_tensor(bg_image, bg_color, width, height, device, dtype)

    import pyvirtualcam
    vcam = pyvirtualcam.Camera(width=width, height=height, fps=fps, device=output_device)
    click.echo(f"[rvm-webcam] virtual camera live on {output_device}")

    use_cuda = device == "cuda"

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
    t_start = time.perf_counter_ns()
    acc_h2d = 0
    acc_infer = 0
    acc_take = 0
    acc_d2h = 0
    acc_sleep = 0
    cur = 0
    prev_buf_idx = None

    def _emit(out_gpu):
        out = out_gpu.cpu().numpy()
        if debug:
            _draw_debug_overlay(out, ema_fps, ema_infer, ema_take, ema_d2h)
        vcam.send(out)

    prev_frame_ts = 0
    ema_fps = 0.0
    ema_infer = 0.0
    ema_take = 0.0
    ema_d2h = 0.0

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
                                t_start = time.perf_counter_ns()
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

                prep = pipeline.prep
                assert prep is not None

                if prev_buf_idx is None:
                    if debug:
                        prev_frame_ts = 0
                    idx, buf = prep.take(timeout=5.0)
                    if buf is None:
                        continue
                    launch_h2d(buf, src_gpu[cur], cur)
                    prev_buf_idx = idx
                    continue

                t0 = time.perf_counter_ns()
                if use_cuda:
                    h2d_events[cur].wait()
                prep.release(prev_buf_idx)
                t1 = time.perf_counter_ns()
                acc_h2d += t1 - t0

                fgr, pha, *rec = model(src_gpu[cur], *rec, downsample_ratio)
                t2 = time.perf_counter_ns()
                acc_infer += t2 - t1

                com = fgr * pha + bg_tensor * (1 - pha)
                out_gpu = (com[0].permute(1, 2, 0).clamp(0, 1) * 255).byte()

                nxt = 1 - cur
                idx, buf = prep.take(timeout=5.0)
                t3 = time.perf_counter_ns()
                acc_take += t3 - t2
                if buf is None:
                    _emit(out_gpu)
                    break
                launch_h2d(buf, src_gpu[nxt], nxt)

                _emit(out_gpu)

                frame_count += 1
                prev_buf_idx = idx
                cur = nxt

                t4 = time.perf_counter_ns()
                vcam.sleep_until_next_frame()
                t5 = time.perf_counter_ns()
                acc_sleep += t5 - t4
                acc_d2h += t4 - t3

                if debug:
                    inst_interval = (t0 - prev_frame_ts) / 1e9 if prev_frame_ts > 0 else 0
                    inst_fps = 1.0 / inst_interval if inst_interval > 0 else 0
                    inst_infer = (t2 - t1) / 1e6
                    inst_take = (t3 - t2) / 1e6
                    inst_d2h = (t4 - t3) / 1e6
                    a = 0.1
                    if prev_frame_ts == 0:
                        ema_fps = inst_fps
                        ema_infer = inst_infer
                        ema_take = inst_take
                        ema_d2h = inst_d2h
                    else:
                        ema_fps = a * inst_fps + (1 - a) * ema_fps
                        ema_infer = a * inst_infer + (1 - a) * ema_infer
                        ema_take = a * inst_take + (1 - a) * ema_take
                        ema_d2h = a * inst_d2h + (1 - a) * ema_d2h
                    prev_frame_ts = t0

                if frame_count % 100 == 0 and frame_count > 0:
                    n = frame_count
                    ms = lambda ns: ns / 1e6
                    click.echo(
                        f"[rvm-webcam] {frame_count / ((t5 - t_start) / 1e9):.1f} fps  "
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
