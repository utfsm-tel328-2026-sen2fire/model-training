{
  pkgs,
  ...
}:
let
  gcc = pkgs.gcc.cc;
  gccVersion = pkgs.gcc.version;
  targetTriplet = pkgs.stdenv.hostPlatform.config;
in
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

  env = {
    # MIOpen JIT-compila los kernels de conv con el clang embebido en
    # libhiprtc.so, que busca los headers de la C++ standard library en
    # /usr/include/c++/<ver>/. Esa ruta no existe en NixOS, así que el
    # compilación aborta con "fatal error: 'type_traits' file not found".
    # CPATH le dice dónde están realmente (vienen con el gcc de nixpkgs).
    # Referencia a pkgs.gcc.cc y no a un hash del store: se mueve solo con
    # el pin de nixpkgs.
    CPATH = "${gcc}/include/c++/${gccVersion}:${gcc}/include/c++/${gccVersion}/${targetTriplet}";
  };
}