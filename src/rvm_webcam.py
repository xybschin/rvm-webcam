#!/usr/bin/env python3
"""rvm-webcam: real-time background removal virtual camera using RobustVideoMatting."""

import json
import os
import signal
import sys
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
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_w, actual_h) != (width, height):
        click.echo(
            f"[rvm-webcam] WARNING: requested {width}x{height} but device reports "
            f"{actual_w}x{actual_h}; frames will be resized.",
            err=True,
        )
    return cap


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

    model = load_model(model_path, backbone, device)

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

    if on_demand:
        cap = None
        wakelock = None
        last_poll = 0.0
        DEBOUNCE_SEC = 0.0
        POLL_INTERVAL = 1.0
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        click.echo(
            f"[rvm-webcam] on-demand enabled: webcam opens when consumer reads {output_device}"
        )
    else:
        cap = open_capture(input_device, width, height, fps)

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

    try:
        with torch.no_grad():
            while running:
                if on_demand:
                    now = time.monotonic()
                    if now - last_poll >= POLL_INTERVAL:
                        last_poll = now
                        consumers = _consumer_count(output_device)

                        if consumers > 0:
                            if cap is None:
                                click.echo(
                                    f"[rvm-webcam] {consumers} consumer(s) connected, opening webcam",
                                    err=True,
                                )
                                try:
                                    cap = open_capture(input_device, width, height, fps)
                                except RuntimeError as e:
                                    click.echo(f"[rvm-webcam] {e}, retrying...", err=True)
                                    cap = None
                                    continue
                                rec = [None] * 4
                                frame_count = 0
                            wakelock = None
                        else:
                            if cap is not None:
                                if wakelock is None:
                                    wakelock = now
                                elif now - wakelock >= DEBOUNCE_SEC:
                                    click.echo(
                                        "[rvm-webcam] no consumers for 2s, releasing webcam",
                                        err=True,
                                    )
                                    cap.release()
                                    cap = None
                                    wakelock = None
                                    rec = [None] * 4

                    if cap is None:
                        vcam.send(black_frame)
                        vcam.sleep_until_next_frame()
                        continue

                ret, frame = cap.read()
                if not ret:
                    click.echo("[rvm-webcam] frame read failed, retrying...", err=True)
                    time.sleep(0.01)
                    continue

                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                src = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype) / 255.0

                with torch.autocast(device_type=device, dtype=dtype, enabled=(device != "cpu")):
                    try:
                        fgr, pha, *rec = model(src, *rec, downsample_ratio)
                    except Exception:
                        if use_compile:
                            click.echo(
                                "[rvm-webcam] WARNING: torch.compile failed at runtime; "
                                "falling back to uncompiled model",
                                err=True,
                            )
                            model = original_model
                            use_compile = False
                            fgr, pha, *rec = model(src, *rec, downsample_ratio)
                        else:
                            raise

                com = fgr * pha + bg_tensor * (1 - pha)
                out = (com[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()

                vcam.send(out)

                if preview:
                    _real_stdout.buffer.write(out.tobytes())
                    _real_stdout.buffer.flush()

                frame_count += 1

                vcam.sleep_until_next_frame()

                if frame_count % 100 == 0 and frame_count > 0:
                    elapsed = time.time() - t_start
                    click.echo(f"[rvm-webcam] {frame_count / elapsed:.1f} fps avg")
    finally:
        if cap is not None:
            cap.release()
        vcam.close()
        click.echo("[rvm-webcam] released camera and virtual device, exiting.")


if __name__ == "__main__":
    main()
