{
pkgs,
...
}:
{
languages.python = {
  enable = true;
  venv.enable = true;
  libraries = [
      # Añadir aquí las librerias que pidan los paquetes instalados con pip
      pkgs.zlib
    ];
};
}
