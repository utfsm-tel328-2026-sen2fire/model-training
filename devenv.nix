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
      numactl    # libnuma.so.1
    ];
  uv = {
    enable = true;
    sync.enable = true;
  };
};
}
