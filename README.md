# rvm-webcam

Real-time background removal virtual camera using [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting).

## Prerequisites

1. **AMD (ROCm)** or **NVIDIA (CUDA)** GPU (CPU fallback works).
2. Load v4l2loopback ([see below](#v4l2loopback-setup)).
3. Download model checkpoint:
   ```sh
   curl -fL -o models/rvm_mobilenetv3.pth \
     https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth
   ```

## Usage

```sh
nix run github:xybschin/rvm-webcam --impure -- --model-path models/rvm_mobilenetv3.pth
```

Options can also be set in `~/.config/rvm-webcam/config.json` (underscores, not hyphens). CLI flags override config.

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | (required) | Path to `.pth` checkpoint |
| `--backbone` | `mobilenetv3` | `mobilenetv3` or `resnet50` |
| `--input-device` | `/dev/video0` | Physical webcam |
| `--output-device` | `/dev/video10` | v4l2loopback virtual device |
| `--width` / `--height` / `--fps` | `1280` / `720` / `30` | Capture resolution and framerate |
| `--downsample-ratio` | `0.25` | Inference resolution fraction (lower = faster) |
| `--bg-color` | `0,255,0` | Background as `R,G,B` |
| `--bg-image` | — | Background image path (JPG/PNG) |
| `--precision` | `auto` | `auto`, `fp16`, or `fp32` |
| `--on-demand` | off | Only open webcam when a consumer reads `/dev/video10` |
| `--debug` | off | Overlay FPS and stage timings on the output frame |

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

## Building

```sh
nix develop --impure         # dev shell with ROCm torch + tooling
nix build . --impure         # standalone binary
./result/bin/rvm-webcam --model-path models/rvm_mobilenetv3.pth
```

## NixOS / home-manager module

```nix
{
  inputs.rvm-webcam.url = "github:xybschin/rvm-webcam";
  # Add rvm-webcam.nixosModules.default or rvm-webcam.homeManagerModules.default

  services.rvm-webcam = {
    enable = true;
    modelPath = "/home/you/models/rvm_mobilenetv3.pth";
    # Optional: backbone, width, height, fps, extraConfig = { bg_color = "0,255,0"; ... }
  };
}
```
