{
  description = "rvm-webcam - background removal virtual camera CLI";

  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
      "https://cache.nixos-cuda.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M="
    ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
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
            description = "Absolute path to the RVM model .pth file";
          };
          backbone = lib.mkOption {
            type = lib.types.enum [ "mobilenetv3" "resnet50" ];
            default = "mobilenetv3";
            description = "RVM backbone architecture";
          };
          width = lib.mkOption { type = lib.types.int; default = 1280; };
          height = lib.mkOption { type = lib.types.int; default = 720; };
          fps = lib.mkOption { type = lib.types.int; default = 30; };
          extraConfig = lib.mkOption {
            type = lib.types.attrsOf lib.types.raw;
            default = { };
            description = "Additional config.json entries (e.g. bg_color, precision)";
          };
        };

        config = lib.mkIf config.services.rvm-webcam.enable {
          environment.systemPackages = [ self.packages.${pkgs.stdenv.hostPlatform.system}.default ];

          systemd.user.services.rvm-webcam = {
            description = "rvm-webcam background removal virtual camera";
            documentation = [ "https://github.com/xybschin/rvm-webcam" ];
            after = [ "graphical-session.target" ];
            wants = [ "graphical-session.target" ];
            wantedBy = [ "graphical-session.target" ];
            serviceConfig = {
              Type = "simple";
              ExecStart = "${self.packages.${pkgs.stdenv.hostPlatform.system}.default}/bin/rvm-webcam --on-demand";
              Restart = "on-failure";
              RestartSec = "3";
            };
          };

          environment.etc."rvm-webcam/config.json".text = builtins.toJSON (
            {
              model_path = config.services.rvm-webcam.modelPath;
              backbone = config.services.rvm-webcam.backbone;
              width = config.services.rvm-webcam.width;
              height = config.services.rvm-webcam.height;
              fps = config.services.rvm-webcam.fps;
            } // config.services.rvm-webcam.extraConfig
          );
        };
      };
      homeManagerModules.default = { config, pkgs, lib, ... }: {
        options.services.rvm-webcam = {
          enable = lib.mkEnableOption "rvm-webcam background removal virtual camera";
          modelPath = lib.mkOption {
            type = lib.types.str;
            description = "Absolute path to the RVM model .pth file";
          };
          backbone = lib.mkOption {
            type = lib.types.enum [ "mobilenetv3" "resnet50" ];
            default = "mobilenetv3";
            description = "RVM backbone architecture";
          };
          width = lib.mkOption { type = lib.types.int; default = 1280; };
          height = lib.mkOption { type = lib.types.int; default = 720; };
          fps = lib.mkOption { type = lib.types.int; default = 30; };
          extraConfig = lib.mkOption {
            type = lib.types.attrsOf lib.types.raw;
            default = { };
            description = "Additional config.json entries (e.g. bg_color, precision)";
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
              ExecStart = "${self.packages.${pkgs.stdenv.hostPlatform.system}.default}/bin/rvm-webcam --on-demand";
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
              backbone = config.services.rvm-webcam.backbone;
              width = config.services.rvm-webcam.width;
              height = config.services.rvm-webcam.height;
              fps = config.services.rvm-webcam.fps;
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
            cudaSupport = true; # CUDA torch-bin is prebuilt on cache.nixos-cuda.org; false yields an uncached build
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

        opencv4-python = pythonPackages.toPythonModule (
          pkgs.opencv4.override {
            enablePython = true;
            pythonPackages = pythonPackages;
          }
        );

        pythonEnv = pkgs.python312.withPackages (
          ps: with ps; [
            torch-bin
            torchvision-bin
            opencv4-python
            numpy
            pyvirtualcam
            click
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
            lspEnv
            pkgs.ruff
            pkgs.pyright
            pkgs.neovim
            pkgs.ffmpeg
            pkgs.v4l-utils
            pkgs.glibc # provides ldconfig needed by torch.compile (Triton)
            pkgs.git # torch.hub.load clones PeterL1n/RobustVideoMatting at runtime
          ];

          # libcuda.so comes from the host NVIDIA driver (/run/opengl-driver/lib on NixOS),
          # not from nixpkgs' nvidia_x11 which must match the running kernel driver exactly.
          shellHook = ''
            export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            echo "rvm-webcam dev shell: python=$(python --version), cuda available via torch.cuda.is_available()"
          '';
        };

        packages.default = pkgs.writeShellApplication {
          name = "rvm-webcam";
          runtimeInputs = [
            pythonEnv
            pkgs.v4l-utils
            pkgs.git # torch.hub.load clones PeterL1n/RobustVideoMatting at runtime
          ];
          text = ''
            export TORCH_HOME="''${TORCH_HOME:-''${XDG_CACHE_HOME:-$HOME/.cache}/torch}"
            export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            exec ${pythonEnv}/bin/python ${./src/rvm_webcam.py} "$@"
          '';
        };

        packages.systemd-unit = pkgs.runCommand "rvm-webcam-systemd-unit" { } ''
          mkdir -p $out/lib/systemd/user
          cat > $out/lib/systemd/user/rvm-webcam.service << EOF
[Unit]
Description=rvm-webcam background removal virtual camera
Documentation=https://github.com/xybschin/rvm-webcam
After=graphical-session.target
Wants=graphical-session.target

[Service]
Type=simple
ExecStart=${self.packages.${system}.default}/bin/rvm-webcam --on-demand
Restart=on-failure
RestartSec=3

[Install]
WantedBy=graphical-session.target
EOF
        '';

      }
    );
}
