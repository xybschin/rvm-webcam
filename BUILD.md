# Building & Integration

> **AMD GPU users (ROCm):** Building PyTorch from source with ROCm support takes a long time. Pre-built ROCm wheels are an alternative — use `--precision fp32` if `torch.compile` isn't stable on your ROCm version.

## Development shell

```sh
nix develop --impure
```

Installs Python with ROCm-enabled torch, torchvision, opencv, pyvirtualcam, plus dev tools (ruff, pyright, neovim).

## Build

```sh
nix build . --impure
./result/bin/rvm-webcam --model-path models/rvm_mobilenetv3.pth
```

## systemd user service (standalone)

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

### Available options (NixOS module)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable` | bool | — | Enable the service |
| `modelPath` | string | — | Absolute path to `.pth` checkpoint |
| `backbone` | `"mobilenetv3"` / `"resnet50"` | `"mobilenetv3"` | |
| `width` / `height` / `fps` | int | `1280` / `720` / `30` | |
| `extraConfig` | attrset | `{}` | Extra `config.json` entries (e.g. `bg_color`, `precision`) |

v4l2loopback must still be configured separately (see README).

## Home-manager module

```nix
{
  imports = [ inputs.rvm-webcam.homeManagerModules.default ];
  services.rvm-webcam = {
    enable = true;
    modelPath = "/home/you/models/rvm_mobilenetv3.pth";
  };
}
```

Same options as the NixOS module, but writes config to `~/.config/rvm-webcam/config.json` instead of `/etc/rvm-webcam/config.json`.

## Implementation

Single file `src/rvm_webcam.py`. Key details:

- **`load_model()`** — Fetches backbone via `torch.hub`, loads state dict.
- **`open_capture()`** — `cv2.VideoCapture` with MJPG fourcc.
- **`_consumer_count()`** — Scans `/proc/*/fd/` for processes holding the output device open.
- **On-demand loop** — Lazy-opens webcam when a consumer appears; releases instantly when the last consumer disconnects. Sends black frames during idle.
- **`config.json`** — Provides defaults for all CLI options. Precedence: CLI > config.json > built-in defaults.
