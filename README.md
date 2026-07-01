# rvm-webcam

Real-time background removal virtual camera powered by [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) (RVM). Captures your webcam feed, removes the background frame-by-frame using a deep neural network on GPU (or CPU fallback), composites a solid-color or image background, and outputs to a [v4l2loopback](https://github.com/umlaeute/v4l2loopback) virtual camera device.

Accepts a pre-trained `.pth` checkpoint (MobiNetV3 or ResNet50 backbone) and exposes a virtual `/dev/videoN` device that any app (Zoom, OBS, browser) can consume.

## Pipeline

```
┌────────────┐   ┌──────────┐   ┌─────────────┐   ┌───────────┐   ┌───────────────┐
│  Webcam    │ → │ BGR→RGB  │ → │  RVM Model  │ → │ Composite │ → │ pyvirtualcam  │
│ /dev/video0│   │  + Norm  │   │  (GPU/CPU)  │   │  fgr*pha  │   │ /dev/video10  │
└────────────┘   └──────────┘   │  fgr + pha  │   │  + bg*(1- │   └───────────────┘
                                └─────────────┘   │   pha)    │
                                                  └───────────┘
```

1. **Capture** — OpenCV reads raw BGR frames from a physical webcam (`/dev/video0`). MJPG codec reduces USB bandwidth at high resolutions.
2. **Preprocess** — Frame is converted BGR→RGB, normalized to `[0,1]`, reshaped to `(1, 3, H, W)` tensor, and moved to the target device/dtype.
3. **Inference** — RVM runs in `torch.no_grad()`, optionally with `autocast` fp16 on CUDA. The model outputs a foreground image (`fgr`) and an alpha matte (`pha`). A 4-element recurrent state buffer (`rec`) enables temporal consistency across frames.
4. **Composite** — Foreground is blended over a configurable background: `com = fgr * pha + bg * (1 - pha)`. The background is either a solid color or a JPEG/PNG image resized to the frame dimensions.
5. **Output** — The composite is clamped, converted RGB→RGBA, sent to `pyvirtualcam` (writes to the v4l2loopback device), and the loop sleeps until the next frame interval.

Downstream apps see the virtual camera as a normal `/dev/video10` with the background already removed.

## Usage

### Prerequisites

1. **Load the v4l2loopback module** on the host (creates `/dev/video10`) — see
   [v4l2loopback setup](#v4l2loopback-host-requirement) below.
2. **Download a model checkpoint** (weights are not committed to this repo):
   ```sh
   mkdir -p models
   curl -fL -o models/rvm_mobilenetv3.pth \
     https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth
   # optional, higher quality:
   curl -fL -o models/rvm_resnet50.pth \
     https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth
   ```
3. **Enter the dev shell** for dependencies (torch, opencv, pyvirtualcam):
   ```sh
   nix develop
   ```

### Run

Run the script directly with `python`:

```sh
# solid-color background (default: green)
python src/rvm_webcam.py \
  --model-path models/rvm_mobilenetv3.pth \
  --backbone mobilenetv3 \
  --input-device /dev/video0 \
  --output-device /dev/video10 \
  --width 1280 --height 720 --fps 30 \
  --downsample-ratio 0.25 \
  --bg-color "0,255,0" \
  --precision auto

# image background
python src/rvm_webcam.py \
  --model-path models/rvm_mobilenetv3.pth \
  --bg-image /path/to/background.jpg

# ResNet50 backbone (higher quality, slower)
python src/rvm_webcam.py \
  --model-path models/rvm_resnet50.pth \
  --backbone resnet50
```

Only `--model-path` is required; all other flags fall back to the defaults listed below.
Press `Ctrl-C` to shut down cleanly (releases the camera and virtual device).

> **First run** fetches the RVM model architecture from GitHub via `torch.hub` (needs `git`
> and internet; the local `.pth` supplies the weights). If you hit GitHub API rate limits,
> export a token: `export GITHUB_TOKEN=<your-token>`. Subsequent runs use the
> `~/.cache/torch/hub/` cache.

Alternatively, `nix build` produces an `rvm-webcam` wrapper you can run in place of
`python src/rvm_webcam.py` (see [Setup](#setup-nix-flake)).

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | (required) | Path to `.pth` checkpoint |
| `--backbone` | `mobilenetv3` | `mobilenetv3` or `resnet50` |
| `--input-device` | `/dev/video0` | Physical webcam device |
| `--output-device` | `/dev/video10` | v4l2loopback virtual device |
| `--width` | `1280` | Frame width |
| `--height` | `720` | Frame height |
| `--fps` | `30` | Target framerate |
| `--downsample-ratio` | `0.25` | Inference resolution fraction (lower = faster, less edge detail) |
| `--bg-color` | `0,255,0` | Composited background as `R,G,B` (mutually exclusive with `--bg-image`) |
| `--bg-image` | — | Path to a background image (JPG, PNG, etc.); resized to frame dimensions (mutually exclusive with `--bg-color`) |
| `--precision` | `auto` | `auto` (fp16 on CUDA, fp32 on CPU), `fp16`, or `fp32` |

## Setup (Nix Flake)

The project provides a Nix flake pinned to `nixos-25.11` with CUDA-enabled `torch-bin`
(CUDA 12.8), `torchvision-bin`, `opencv`, `pyvirtualcam`, and Python tooling. The prebuilt
CUDA binaries are served from `cache.nixos-cuda.org`, so no local compilation is required.

```sh
# Enter dev shell with all dependencies
nix develop

# Build the CLI package (wraps python + deps into a single script)
nix build
```

The `devShell` includes Neovim with pylsp/ruff formatting, `ffmpeg`, `v4l-utils`, and `git`
(needed by `torch.hub` to fetch the RVM backbone at first run).

### v4l2loopback (host requirement)

The virtual camera device is provided by the [v4l2loopback](https://github.com/umlaeute/v4l2loopback)
kernel module, which **cannot** be supplied by a dev shell — kernel modules are loaded on the
host. Set it up out-of-band:

- **NixOS:** add to your system config, then rebuild:
  ```nix
  boot.extraModulePackages = [ config.boot.kernelPackages.v4l2loopback ];
  boot.kernelModules = [ "v4l2loopback" ];
  boot.extraModprobeConfig = ''
    options v4l2loopback devices=1 video_nr=10 card_label="rvm-webcam" exclusive_caps=1
  '';
  ```
- **Other distros:** install the `v4l2loopback` package (or DKMS), then:
  ```sh
  sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="rvm-webcam" exclusive_caps=1
  ```

This creates `/dev/video10`, the default `--output-device`. Verify with `v4l2-ctl --list-devices`.

GPU note: CUDA torch loads `libcuda.so` from the host NVIDIA driver
(`/run/opengl-driver/lib` on NixOS). Both the dev shell and the built package prepend this
path to `LD_LIBRARY_PATH` automatically. Falls back to CPU if no GPU is present.

## Implementation

The entire pipeline is a single file: `src/rvm_webcam.py` (~125 lines). Key details:

- **`load_model()`** — Downloads the backbone from `PeterL1n/RobustVideoMatting` via `torch.hub`, loads the state dict.
- **`open_capture()`** — Wraps `cv2.VideoCapture` with resolution, FPS, and MJPG fourcc.
- **Signal handling** — `SIGINT`/`SIGTERM` gracefully tear down: releases the physical camera and closes the virtual device.
- **Per-frame stats** — Logs average FPS every 100 frames.
