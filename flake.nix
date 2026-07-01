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
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (
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

        pythonEnv = pkgs.python312.withPackages (
          ps: with ps; [
            torch-bin
            torchvision-bin
            opencv4
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
            # torch.hub caches the cloned RVM repo here; default ~/.cache/torch is fine
            export TORCH_HOME="''${TORCH_HOME:-''${XDG_CACHE_HOME:-$HOME/.cache}/torch}"
            # libcuda.so ships with the host NVIDIA driver, not nixpkgs. On NixOS it lives
            # at /run/opengl-driver/lib; prepend it so CUDA torch can find the driver.
            export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            exec ${pythonEnv}/bin/python ${./src/rvm_webcam.py} "$@"
          '';
        };
      }
    );
}
