# model-training

Entrenamiento de un U-Net para segmentación de fuego con imágenes proporcionadas por el satélite Sentinel-2.

# Recursos
  https://browser.dataspace.copernicus.eu/
  https://zenodo.org/records/13384289
  https://zenodo.org/records/10881058
## Setup

- Python 3.12, [uv](https://docs.astral.sh/uv/)
- `uv sync` instala todo (incluye el paquete en modo editable)

## Variables de entorno (en `.env`)

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `SEN2FIRE_ROOT` | sí | — | Ruta al dataset (NFS o local) con `scene1/`, `scene2/`, … |
| `SEN2FIRE_LOCAL_CACHE` | no | sin caché | Ruta local donde clonar el dataset la primera vez |

## Uso

```bash
# smoke test del loader (carga dataset, imprime stats, prueba 1 batch)
python main.py

# entrenamiento mínimo (1 epoch, batch_size 1, 4 train + 2 val samples)
python train.py --epochs 1 --batch-size 1 --num-workers 0 --limit-train 4 --limit-val 2

# entrenamiento real
python train.py --epochs 10 --batch-size 2 --num-workers 2
```

## Estructura

```
src/sen2fire/
  data/    config, cache, dataset
  models/  U-Net, Dice + BCE
main.py    smoke test del loader
train.py   loop de entrenamiento
```
