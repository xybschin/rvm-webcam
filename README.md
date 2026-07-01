# rvm-webcam

Real-time background removal virtual camera using [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting). Captures webcam, removes background via GPU (or CPU), composites a color/image background, outputs to a v4l2loopback device.

## Prerequisites

1. Load v4l2loopback — see [setup](#v4l2loopback-setup) below.
2. Download a model checkpoint:
   ```sh
   curl -fL -o models/rvm_mobilenetv3.pth \
     https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth
   ```

## Usage

```sh
nix run github:xybschin/rvm-webcam --impure -- --model-path models/rvm_mobilenetv3.pth
```

Only `--model-path` is required; all other flags have defaults. Press `Ctrl-C` to clean up.

> First run fetches the RVM architecture via `torch.hub` (needs `git` + internet). Subsequent runs use `~/.cache/torch/hub/`.

Preview to ffplay (no display server needed):

```sh
nix run github:xybschin/rvm-webcam --impure -- --model-path models/rvm_mobilenetv3.pth --preview \
  | ffplay -f rawvideo -pixel_format rgb24 -video_size 1280x720 -i -
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | (required) | Path to `.pth` checkpoint |
| `--backbone` | `mobilenetv3` | `mobilenetv3` or `resnet50` |
| `--input-device` | `/dev/video0` | Physical webcam |
| `--output-device` | `/dev/video10` | v4l2loopback virtual device |
| `--width` | `1280` | Frame width |
| `--height` | `720` | Frame height |
| `--fps` | `30` | Target framerate |
| `--downsample-ratio` | `0.25` | Inference resolution fraction (lower = faster) |
| `--bg-color` | `0,255,0` | Background as `R,G,B` (mutually exclusive with `--bg-image`) |
| `--bg-image` | — | Background image path (JPG/PNG) |
| `--compile` | off | `torch.compile` (PyTorch ≥ 2.0, ~10-30% on CUDA) |
| `--precision` | `auto` | `auto`, `fp16`, or `fp32` |
| `--preview` | off | Pipe raw RGB24 to stdout for ffplay |
| `--on-demand` | off | Only open webcam when a consumer reads `/dev/video10` |

All options can also be set in `~/.config/rvm-webcam/config.json` (underscores, not hyphens). CLI flags override config.

## v4l2loopback setup

- **NixOS:** add to system config:
  ```nix
  boot.extraModulePackages = [ config.boot.kernelPackages.v4l2loopback ];
  boot.kernelModules = [ "v4l2loopback" ];
  boot.extraModprobeConfig = ''
    options v4l2loopback devices=1 video_nr=10 card_label="rvm-webcam" exclusive_caps=1
  '';
  ```
- **Other distros:** `sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="rvm-webcam" exclusive_caps=1`

Creates `/dev/video10`. Verify with `v4l2-ctl --list-devices`.
