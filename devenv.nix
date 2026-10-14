{
pkgs,
...
}:
{
languages.python = {
  enable = true;
  version = "3.12";
  venv.enable = true;
  libraries = [
      # Añadir aquí las librerias que pidan los paquetes instalados con uv
      pkgs.zlib
    ];
  uv = {
    enable = true;
    sync.enable = true;
  };
};
}
