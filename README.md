# rvm-webcam

Real-time background removal virtual camera using RobustVideoMatting on AMD ROCm.

Drive a [v4l2loopback](https://github.com/umlaeute/v4l2loopback) device with a GPU-accelerated matting pipeline — zero-copy host-device transfers via `hipHostRegister`, native FP16 ONNX inference via MIGraphX, and CPU-side FP32 preprocessing to avoid hardware-emulated float16 bottlenecks. The physical camera is only opened while a real consumer (browser tab, video call, OBS, etc.) is actually reading from the virtual camera.

## Architecture

```
ffmpeg (mjpeg→rgb24) → FrameGrabber → FramePrep (CPU FP32) ──[H2D: FP32→FP16 copyto]──→ RVM ONNX (GPU FP16, MIGraphX)
                                                                                             ↕ recurrent r1-r4 states (static shape)
                                                                                          ←──[D2H: HIP-pinned DMA]──┐
                                                                                                                    ↓
                                                                                                 Compositor (FP32 clip/astype) → vcam.send(uint8)
```

Gated by `LoopbackConsumerMonitor`: the virtual camera (`vcam`) always streams (idle frames when no consumer is attached, so the v4l2loopback OUTPUT token stays claimed and consumers can connect at all), but the physical camera capture (`ffmpeg`) and GPU inference only run while `V4L2_EVENT_PRI_CLIENT_USAGE` reports an active CAPTURE-side consumer — i.e. the "shutter" only opens when something is actually watching.

- **Consumer-gated capture**: physical camera and GPU inference are only active while a real consumer is attached to the virtual camera device (see [Consumer Detection](#consumer-detection)).
- **Precision isolation**: Preprocessing/scaling in CPU FP32; single `np.copyto` downcast to FP16 at the GPU boundary.
- **HIP DMA pinning**: `hipHostRegister` on `mmap`-backed memory eliminates staging copies.
- **Static recurrent-state shapes**: RVM's r1-r4 state tensors are pre-allocated at their final spatial shape (not a broadcastable placeholder) so MIGraphX JIT-compiles its graph exactly once instead of recompiling (which can hang for minutes) when the shape changes between frame 1 and frame 2.
- **Hard startup failures**: `RuntimeError` if GPU execution providers (`MIGraphXExecutionProvider`) are absent — no silent CPU fallback.
- **Lifecycle safety**: `PinnedBuffer` wrapper with `cleanup()` → `hipHostUnregister` + `munlock` + `mmap.close()`; SIGINT/SIGTERM set a `running` flag so the main loop exits and runs its `finally` teardown (physical camera release, virtual camera close, pinned buffer cleanup) rather than force-killing the process.

## Consumer Detection

The physical camera is not opened at process start. Instead, `rvm-webcam` subscribes to v4l2loopback's `V4L2_EVENT_PRI_CLIENT_USAGE` event on the output device — the same private event browsers and OBS use to show a "camera in use" indicator only while something is actually reading frames. The event fires whenever a CAPTURE-side consumer calls `VIDIOC_STREAMON`/`STREAMOFF`.

- No consumer attached: virtual camera keeps streaming a solid background-color idle frame (required so the v4l2loopback OUTPUT token stays claimed — otherwise a consumer's `VIDIOC_STREAMON` fails with `-EIO`); physical camera and GPU inference are idle.
- Consumer attaches (e.g. `ffplay /dev/video10`, a browser tab, a video call): physical camera opens, GPU pipeline starts producing real matted frames.
- Consumer disconnects: physical camera and GPU pipeline shut down again.

If the underlying ioctl/event isn't supported (non-Linux, non-v4l2loopback backend, or an old kernel module version), detection fails open — the pipeline always runs, matching the previous always-on behavior.

## Quick Start (Nix)

```shell
# Dev shell with all dependencies
nix develop

# Run the environment checker
nix run .#check-environment

# Run the pipeline (requires rvm_mobilenetv3_fp16.onnx and v4l2loopback)
sudo modprobe v4l2loopback
nix run . -- --model-path /path/to/model.onnx

# Run tests
nix develop -c pytest src/test_pipeline.py -v
```

## CLI

```text
Usage: rvm-webcam [OPTIONS]

Options:
  --model-path PATH         ONNX model file
  --input-device TEXT       Webcam device  [default: /dev/video0]
  --output-device TEXT      v4l2loopback device  [default: /dev/video10]
  --width INTEGER           Frame width  [default: 1280]
  --height INTEGER          Frame height  [default: 720]
  --fps INTEGER             Target framerate  [default: 30]
  --downsample-ratio FLOAT  Inference resolution scale  [default: 0.25]
  --device-id INTEGER       GPU device index  [default: 0]
  --cache-dir PATH          MIGraphX .mxr compile cache directory (avoids
                             recompilation on restart; without it, every
                             process start pays a one-time JIT compile of
                             roughly one to a few minutes)
```

Config can also be placed at `~/.config/rvm-webcam/config.json` or `/etc/rvm-webcam/config.json`; CLI flags override file values.

## Testing

Mock-based unit tests that don't require a camera or GPU (plus two live tests against a real v4l2loopback device that skip gracefully if unavailable):

```shell
nix develop -c pytest src/test_pipeline.py -v
```

Six test groups: pinned memory lifecycle (`PinnedBuffer`), model signature validation (`_validate_model_signature`), execution provider checks (`RecurrentStateBufferGPU` / `MIGraphXSession`), worker thread exception propagation (`error_q`), HIP runtime loader fallback (`_load_hip_runtime`), and loopback consumer detection (`LoopbackConsumerMonitor` — includes a live end-to-end test with a real ffmpeg consumer).

## Environment Checks

```shell
nix run .#check-environment
```

Verifies `/dev/video*` permissions, `v4l2loopback` module, ROCm driver (`/dev/kfd`, `rocm-smi`, `libamdhip64.so`), `ffmpeg`, and ONNX Runtime providers.

## Files

| File | Purpose |
|---|---|
| `src/rvm_webcam.py` | Pipeline: HIP-pinned buffers, `FrameGrabber`/`FramePrep` threads, `MIGraphXSession`, `RecurrentStateBufferGPU`, `LoopbackConsumerMonitor`, compositor, CLI entry point |
| `src/test_pipeline.py` | Pytest suite (mostly no hardware required; two tests exercise a real v4l2loopback device and skip if unavailable) |
| `src/check_environment.py` | Dependency verification CLI |
| `scripts/patch_onnx_resize.py` | Standalone script to patch RVM ONNX models for MIGraphX compatibility (Resize `linear`→`nearest`, explicit `AveragePool` `count_include_pad`) |
| `flake.nix` | Nix flake: dev shell, packages, NixOS/Home Manager modules |

## NixOS Module

```nix
services.rvm-webcam = {
  enable = true;
  modelPath = "/path/to/rvm_mobilenetv3_fp16.onnx";
  # optional overrides:
  # width = 1920; height = 1080; fps = 60;
  # cacheDir = "/var/cache/rvm-webcam/migraphx"; # default; auto-provisioned, video-group writable
  # extraConfig = { downsample_ratio = 0.5; device_id = 1; };
};

# Required: the user running the service must be able to open the physical
# camera and GPU render node.
users.users.<name>.extraGroups = [ "video" "render" ];
```

The service is registered as `systemd.user.services.rvm-webcam`, started under `graphical-session.target` for the logged-in user (not a system-wide root service). `cacheDir`'s directory is provisioned via `systemd.tmpfiles.rules` (mode `0775`, group `video`) since `/var/cache` itself is not writable by unprivileged users.

## Home Manager Module

```nix
{
  imports = [ rvm-webcam.homeManagerModules.default ];
  services.rvm-webcam = {
    enable = true;
    modelPath = "/path/to/rvm_mobilenetv3_fp16.onnx";
    # cacheDir defaults to null (disabled) here; point it at a path under
    # XDG_CACHE_HOME to persist the MIGraphX compile cache, e.g.:
    # cacheDir = "${config.xdg.cacheHome}/rvm-webcam/migraphx";
  };
}
```

Unlike the NixOS module, this only manages a per-user systemd unit and `xdg.configFile` — it does not touch `boot.kernelModules` or system packages, so `v4l2loopback` must already be loaded (e.g. via the NixOS module on the same host, or `sudo modprobe v4l2loopback` / `/etc/modules-load.d/`) and the user must already be in the `video`/`render` groups.

## Dependencies

- Python 3.12, ONNX Runtime (with MIGraphX execution provider), `onnx`, NumPy, Click, Pillow, pyvirtualcam
- `ffmpeg` (physical camera capture pipeline) and `v4l-utils` (`v4l2-ctl` for format negotiation)
- ROCm stack: `rocmPackages.clr` (provides `libamdhip64.so` for HIP DMA)
- Kernel: `v4l2loopback` + V4L2 device for the camera
