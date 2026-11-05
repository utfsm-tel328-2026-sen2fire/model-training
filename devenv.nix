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
      gcc.lib       # libgomp.so.1
      numactl       # libnuma.so.1
    ];
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

env = {
    # MIOpen JIT-compila los kernels de conv con el clang embebido en
    # libhiprtc.so, que busca los headers de la C++ standard library y de libc
    # en /usr/include/c++/<ver>/ y /usr/include/. Esas rutas no existen en
    # NixOS, así que la compilación aborta con "fatal error: 'type_traits'
    # file not found" y luego con "'features.h' file not found".
    # CPATH le dice dónde están realmente: los headers de C++ vienen con gcc,
    # los de libc con glibc.
    # Referencia a pkgs.gcc.cc y pkgs.glibc.dev, no a hashes del store: se
    # mueven solos con el pin de nixpkgs.
    CPATH = "${gcc}/include/c++/${gccVersion}:${gcc}/include/c++/${gccVersion}/${targetTriplet}:${pkgs.glibc.dev}/include";
  };
}
