{
  pkgs,
  ...
}:
{
  languages.python = {
    enable = true;
    version = "3.12";
    venv.enable = true;
    libraries = with pkgs; [
      # Añadir aquí las librerias que pidan los paquetes instalados con uv
      zlib
      zstd          # libzstd.so.1
      gcc.cc.lib    # libgomp.so.1
      numactl       # libnuma.so.1
    ];
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  environment.variables = {
    # MIOpen no puede JIT-compilar kernels de conv en NixOS: el clang embebido
    # en libhiprtc.so busca los headers de la C++ standard library en
    # /usr/include/c++/<ver>/, que no existe (viven en el nix store), y aborta
    # con "fatal error: 'type_traits' file not found" -> core dump.
    # FAST saltea el JIT: MIOpen elige un kernel precompilado por heurística.
    # Inocuo en NVIDIA/CPU (no hay MIOpen ahí).
    MIOPEN_FIND_MODE = "FAST";
  };
}
