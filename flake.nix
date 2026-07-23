{
  description = "rvm-webcam - background removal virtual camera CLI (ROCm / MIGraphX)";

  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    {
      nixosModules.default = { config, pkgs, lib, ... }: {
        options.services.rvm-webcam = {
          enable = lib.mkEnableOption "rvm-webcam background removal virtual camera";
          modelPath = lib.mkOption {
            type = lib.types.str;
            description = "Absolute path to the RVM ONNX model (.onnx)";
          };
          width = lib.mkOption { type = lib.types.int; default = 1280; };
          height = lib.mkOption { type = lib.types.int; default = 720; };
          fps = lib.mkOption { type = lib.types.int; default = 30; };
          cacheDir = lib.mkOption {
            type = lib.types.nullOr lib.types.str;
            default = "/var/cache/rvm-webcam/migraphx";
            description = ''
              MIGraphX .mxr compile cache directory (persists compiled model
              across restarts, avoiding a ~2 minute recompile on every
              service start). Since this service runs as a per-user
              `systemd.user.services` unit (not root), this module
              provisions the directory via `systemd.tmpfiles.rules` so it
              is writable by any member of the `video` group.
            '';
          };
          extraConfig = lib.mkOption {
            type = lib.types.attrsOf lib.types.raw;
            default = { };
            description = "Additional config.json entries (e.g. downsample_ratio, device_id)";
          };
        };

        config = lib.mkIf config.services.rvm-webcam.enable {
          boot.kernelModules = [ "v4l2loopback" ];
          boot.extraModprobeConfig = ''
            options v4l2loopback exclusive_caps=0 video_nr=10 card_label="RVM Webcam"
          '';

          environment.systemPackages = [ self.packages.${pkgs.stdenv.hostPlatform.system}.default ];

          # Cameras (/dev/video*) and GPU render nodes (/dev/dri/render*,
          # /dev/kfd) are group-owned by `video`/`render`; the invoking
          # user must be a member of both for the pipeline to open them.
          # This is NOT enforced automatically -- add the user manually:
          #   users.users.<name>.extraGroups = [ "video" "render" ];

          # `systemd.user.services` run as the unprivileged user, so the
          # default /var/cache/rvm-webcam/migraphx (root:root 0755) must
          # be pre-created and made writable, or MIGraphXSession's
          # `cache_path.mkdir()` will fail with a permission error.
          systemd.tmpfiles.rules = lib.optional (config.services.rvm-webcam.cacheDir != null) (
            "d ${config.services.rvm-webcam.cacheDir} 0775 root video - -"
          );

          systemd.user.services.rvm-webcam = {
            description = "rvm-webcam background removal virtual camera";
            documentation = [ "https://github.com/xybschin/rvm-webcam" ];
            after = [ "graphical-session.target" ];
            wants = [ "graphical-session.target" ];
            wantedBy = [ "graphical-session.target" ];
            serviceConfig = {
              Type = "simple";
              ExecStart = "${self.packages.${pkgs.stdenv.hostPlatform.system}.default}/bin/rvm-webcam";
              Restart = "on-failure";
              RestartSec = "3";
            };
          };

          environment.etc."rvm-webcam/config.json".text = builtins.toJSON (
            {
              model_path = config.services.rvm-webcam.modelPath;
              width = config.services.rvm-webcam.width;
              height = config.services.rvm-webcam.height;
              fps = config.services.rvm-webcam.fps;
              cache_dir = config.services.rvm-webcam.cacheDir;
            } // config.services.rvm-webcam.extraConfig
          );
        };
      };
      homeManagerModules.default = { config, pkgs, lib, ... }: {
        options.services.rvm-webcam = {
          enable = lib.mkEnableOption "rvm-webcam background removal virtual camera";
          modelPath = lib.mkOption {
            type = lib.types.str;
            description = "Absolute path to the RVM ONNX model (.onnx)";
          };
          width = lib.mkOption { type = lib.types.int; default = 1280; };
          height = lib.mkOption { type = lib.types.int; default = 720; };
          fps = lib.mkOption { type = lib.types.int; default = 30; };
          cacheDir = lib.mkOption {
            type = lib.types.nullOr lib.types.str;
            default = null;
            description = ''
              MIGraphX .mxr compile cache directory (persists compiled model
              across restarts).  Set to a path under XDG_CACHE_HOME to enable,
              e.g. "rvm-webcam/migraphx".
            '';
          };
          extraConfig = lib.mkOption {
            type = lib.types.attrsOf lib.types.raw;
            default = { };
            description = "Additional config.json entries (e.g. downsample_ratio, device_id)";
          };
        };

        config = lib.mkIf config.services.rvm-webcam.enable {
          home.packages = [ self.packages.${pkgs.stdenv.hostPlatform.system}.default ];

          systemd.user.services.rvm-webcam = {
            Unit = {
              Description = "rvm-webcam background removal virtual camera";
              Documentation = "https://github.com/xybschin/rvm-webcam";
              After = [ "graphical-session.target" ];
              Wants = [ "graphical-session.target" ];
            };
            Service = {
              Type = "simple";
              ExecStart = "${self.packages.${pkgs.stdenv.hostPlatform.system}.default}/bin/rvm-webcam";
              Restart = "on-failure";
              RestartSec = "3";
            };
            Install = {
              WantedBy = [ "graphical-session.target" ];
            };
          };

          xdg.configFile."rvm-webcam/config.json".text = builtins.toJSON (
            {
              model_path = config.services.rvm-webcam.modelPath;
              width = config.services.rvm-webcam.width;
              height = config.services.rvm-webcam.height;
              fps = config.services.rvm-webcam.fps;
              cache_dir = config.services.rvm-webcam.cacheDir;
            } // config.services.rvm-webcam.extraConfig
          );
        };
      };
    }
    // flake-utils.lib.eachSystem [ "x86_64-linux" ] (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            rocmSupport = true;
          };
        };

        pythonPackages = pkgs.python312Packages;

        pyvirtualcam = pythonPackages.buildPythonPackage rec {
          pname = "pyvirtualcam";
          version = "0.15.0";
          format = "wheel";
          src = pkgs.fetchurl {
            url = "https://files.pythonhosted.org/packages/35/51/a9aaa25662e8210fe98f62e4997a9460aea3ce969647b1695d80fd785f62/pyvirtualcam-0.15.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl";
            hash = "sha256-GBhXdi/O/jDekxX++6IT4jsNh6PlyWYvqanMJTnIBDI=";
          };
          propagatedBuildInputs = [ pythonPackages.numpy ];
        };

        patchedOnnxruntime = pythonPackages.onnxruntime.overrideAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ [ pkgs.onnxruntime ];
          postInstall = (old.postInstall or "") + ''
            ln -sf ${pkgs.onnxruntime}/lib/libonnxruntime_providers_migraphx.so \
              "$out/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime_providers_migraphx.so"
          '';
        });

        pythonEnv = pkgs.python312.withPackages (
          ps: with ps; [
            patchedOnnxruntime
            pillow
            numpy
            pyvirtualcam
            click
            ps.onnx
          ]
        );

        testEnv = pkgs.python312.withPackages (
          ps: with ps; [
            pytest
            numpy
            patchedOnnxruntime
            pillow
            click
            pyvirtualcam
            ps.onnx
          ]
        );

        lspEnv = pkgs.python312.withPackages (
          ps: with ps; [
            python-lsp-server
            python-lsp-ruff
          ]
        );

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            testEnv
            lspEnv
            pkgs.ruff
            pkgs.pyright
            pkgs.neovim
            pkgs.ffmpeg
            pkgs.v4l-utils
            pkgs.glibc
            pkgs.git
            pkgs.rocmPackages.clr
          ];

          shellHook = ''
            export ROCM_PATH="${pkgs.rocmPackages.clr}"
            export LD_LIBRARY_PATH="${pkgs.rocmPackages.clr}/lib:${pkgs.onnxruntime}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            echo "rvm-webcam dev shell: python=$(python --version) rocm=${pkgs.rocmPackages.clr.version}"
            echo "run 'pytest src/test_pipeline.py -v' to validate the pipeline"
          '';
        };

        packages.default = pkgs.writeShellApplication {
          name = "rvm-webcam";
          runtimeInputs = [
            pythonEnv
            pkgs.v4l-utils
            pkgs.git
            pkgs.rocmPackages.clr
          ];
          text = ''
            export ROCM_PATH="${pkgs.rocmPackages.clr}"
            export LD_LIBRARY_PATH="${pkgs.rocmPackages.clr}/lib:${pkgs.onnxruntime}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export PYTHONPATH="${./src}:''${PYTHONPATH:+:$PYTHONPATH}"
            exec ${pythonEnv}/bin/python ${./src/rvm_webcam.py} "$@"
          '';
        };

        packages.check-environment = pkgs.writeShellApplication {
          name = "rvm-webcam-check-environment";
          runtimeInputs = [
            pythonEnv
            pkgs.rocmPackages.clr
          ];
          text = ''
            export ROCM_PATH="${pkgs.rocmPackages.clr}"
            export LD_LIBRARY_PATH="${pkgs.rocmPackages.clr}/lib:${pkgs.onnxruntime}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export PYTHONPATH="${./src}:''${PYTHONPATH:+:$PYTHONPATH}"
            exec ${pythonEnv}/bin/python ${./src/check_environment.py} "$@"
          '';
        };

      }
    );
}
