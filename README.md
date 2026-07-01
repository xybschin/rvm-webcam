# rvm-webcam

Real-time background removal virtual camera using [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting). Captures webcam, removes background via GPU (or CPU), composites a color/image background, outputs to a v4l2loopback device.

## Usage

### Prerequisites

1. Load v4l2loopback — see [v4l2loopback setup](#v4l2loopback-setup) below.
2. Download a model checkpoint:
   ```sh
   curl -fL -o models/rvm_mobilenetv3.pth \
     https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth
   ```
3. Enter dev shell: `nix develop`

### Run

```sh
nix run .# --impure -- --model-path models/rvm_mobilenetv3.pth
```

Or build once and run anytime:

```sh
nix build . --impure
./result/bin/rvm-webcam --model-path models/rvm_mobilenetv3.pth
```

Only `--model-path` is required; all other flags have defaults. Press `Ctrl-C` to clean up.

> First run fetches the RVM architecture via `torch.hub` (needs `git` + internet). Subsequent runs use `~/.cache/torch/hub/`.

Preview to ffplay (no display server needed):

```sh
nix run .# --impure -- --model-path models/rvm_mobilenetv3.pth --preview \
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

## systemd user service (standalone)

Run as a background daemon that only captures webcam when a consumer is connected:

```sh
nix build .#systemd-unit --impure
mkdir -p ~/.config/systemd/user
ln -s "$(realpath result/lib/systemd/user/rvm-webcam.service)" \
  ~/.config/systemd/user/rvm-webcam.service
systemctl --user daemon-reload
systemctl --user enable --now rvm-webcam
```

Create `~/.config/rvm-webcam/config.json`:
```json
{
  "model_path": "/home/you/models/rvm_mobilenetv3.pth",
  "width": 1280,
  "height": 720,
  "fps": 30
}
```

## NixOS flake integration

Add as a flake input and use the NixOS module:

```nix
# flake.nix
{
  inputs.rvm-webcam.url = "github:xybschin/rvm-webcam";

  outputs = { self, nixpkgs, rvm-webcam, ... }: {
    nixosConfigurations.mybox = nixpkgs.lib.nixosSystem {
      specialArgs = { inherit rvm-webcam; };
      modules = [
        rvm-webcam.nixosModules.default
        ({ config, ... }: {
          services.rvm-webcam = {
            enable = true;
            modelPath = "/home/you/models/rvm_mobilenetv3.pth";
          };
        })
      ];
    };
  };
}
```

This installs the binary, creates the systemd user service (runs with `--on-demand`), and generates `/etc/rvm-webcam/config.json` from the module options.

Available options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable` | bool | — | Enable the service |
| `modelPath` | string | — | Absolute path to `.pth` checkpoint |
| `backbone` | `"mobilenetv3"` / `"resnet50"` | `"mobilenetv3"` | RVM backbone |
| `width` | int | `1280` | Frame width |
| `height` | int | `720` | Frame height |
| `fps` | int | `30` | Target framerate |
| `extraConfig` | attrset | `{}` | Additional config.json entries (e.g. `bg_color`, `precision`) |

v4l2loopback must still be configured separately in your NixOS config (see [v4l2loopback setup](#v4l2loopback-setup)).

## Implementation

Single file `src/rvm_webcam.py` (~310 lines). Key details:

- **`load_model()`** — Fetches backbone via `torch.hub`, loads state dict.
- **`open_capture()`** — `cv2.VideoCapture` with MJPG fourcc.
- **`_consumer_count()`** — Scans `/proc/*/fd/` for processes holding the output device open.
- **On-demand loop** — Lazy-opens webcam when a consumer appears; releases instantly when the last consumer disconnects. Sends black frames during idle.
- **`config.json`** — Provides defaults for all CLI options. Precedence: CLI > config.json > built-in defaults.
