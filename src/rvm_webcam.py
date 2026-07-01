#!/usr/bin/env python3
"""rvm-webcam: real-time background removal virtual camera using RobustVideoMatting."""

import signal
import time

import click
import cv2
import torch


def load_model(model_path: str, backbone: str, device: str):
    model = torch.hub.load("PeterL1n/RobustVideoMatting", backbone).eval().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def open_capture(device: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device)
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


@click.command()
@click.option("--model-path", required=True, type=click.Path(exists=True), help="Path to rvm_mobilenetv3.pth")
@click.option("--backbone", default="mobilenetv3", type=click.Choice(["mobilenetv3", "resnet50"]))
@click.option("--input-device", default="/dev/video0", help="Real webcam device")
@click.option("--output-device", default="/dev/video10", help="v4l2loopback virtual device")
@click.option("--width", default=1280, type=int)
@click.option("--height", default=720, type=int)
@click.option("--fps", default=30, type=int)
@click.option("--downsample-ratio", default=0.25, type=float, help="Lower = faster, less edge detail")
@click.option("--bg-color", default=None, help="R,G,B for composite background (mutually exclusive with --bg-image)")
@click.option("--bg-image", default=None, type=click.Path(exists=True), help="Path to a background image, e.g. JPG/PNG (mutually exclusive with --bg-color)")
@click.option("--precision", default="auto", type=click.Choice(["auto", "fp16", "fp32"]))
@click.option("--compile", "use_compile", is_flag=True, default=False, help="Apply torch.compile (requires PyTorch ≥ 2.0, gains ~10-30% on CUDA)")
def main(model_path, backbone, input_device, output_device, width, height, fps,
         downsample_ratio, bg_color, bg_image, precision, use_compile):
    if bg_color and bg_image:
        raise click.UsageError("--bg-color and --bg-image are mutually exclusive.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision == "auto":
        dtype = torch.float16 if device == "cuda" else torch.float32
    else:
        dtype = torch.float16 if precision == "fp16" else torch.float32

    click.echo(f"[rvm-webcam] device={device} dtype={dtype} downsample_ratio={downsample_ratio}")

    model = load_model(model_path, backbone, device)

    if use_compile:
        if not hasattr(torch, "compile"):
            click.echo("[rvm-webcam] WARNING: torch.compile not available (requires PyTorch >= 2.0), skipping.", err=True)
        else:
            model = torch.compile(model, mode="reduce-overhead")
            click.echo("[rvm-webcam] torch.compile enabled (reduce-overhead)")

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

    cap = open_capture(input_device, width, height, fps)

    import pyvirtualcam
    vcam = pyvirtualcam.Camera(width=width, height=height, fps=fps, device=output_device)
    click.echo(f"[rvm-webcam] virtual camera live on {output_device}")

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
                ret, frame = cap.read()
                if not ret:
                    click.echo("[rvm-webcam] frame read failed, retrying...", err=True)
                    time.sleep(0.01)
                    continue

                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                src = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype) / 255.0

                with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                    fgr, pha, *rec = model(src, *rec, downsample_ratio)

                com = fgr * pha + bg_tensor * (1 - pha)
                out = (com[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()

                vcam.send(out)
                vcam.sleep_until_next_frame()

                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - t_start
                    click.echo(f"[rvm-webcam] {frame_count / elapsed:.1f} fps avg")
    finally:
        cap.release()
        vcam.close()
        click.echo("[rvm-webcam] released camera and virtual device, exiting.")


if __name__ == "__main__":
    main()
