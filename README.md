# model-training

Entrenamiento de un U-Net para segmentación de fuego con imágenes proporcionadas por el satélite Sentinel-2.

## Setup

[uv](https://docs.astral.sh/uv/) gestiona el entorno: instala Python y las
dependencias (incluido el paquete en modo editable).

Elige el backend de PyTorch según tu GPU:

| GPU              | Comando                  | Wheel que se instala                       |
| ---------------- | ------------------------ | ------------------------------------------ |
| NVIDIA           | `uv sync --extra cuda`   | `torch==2.11.0+cu128`  (último disponible) |
| AMD (ROCm ≥ 7.1) | `uv sync --extra rocm`   | `torch==2.13.0+rocm7.1` (último disponible) |


`uv.lock` incluye las dos variantes resueltas; se regenera con
`uv lock --refresh` si cambian las dependencias o los índices de PyTorch.

`pytest` viene en el grupo `dev`, que `uv sync` instala siempre. Si solo
querés agregar las herramientas de desarrollo sin volver a fijar el backend de
torch (que sí o sí se desinstalaría con un `uv sync` pelado, porque torch solo
está en los extras), usá:

```bash
uv sync --inexact    # instala pytest y no toca tu torch
```

## Variables de entorno (en `.env`)

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `DATASET` | no | `sen2fire` | Qué dataset del registro cargar |
| `SEN2FIRE_ROOT` | sí | — | Ruta al dataset (NFS o local) con `scene1/`, `scene2/`, … |
| `SEN2FIRE_LOCAL_CACHE` | no | sin caché | Si se define, el dataset se clona una vez a esa ruta local |

Las variables reales del entorno pisan a las del `.env`, así que podés
sobrescribir sin editar el archivo:

```bash
SEN2FIRE_LOCAL_CACHE= python train.py    # entrena leyendo directo de la NFS
```

### Caché local

Con `SEN2FIRE_LOCAL_CACHE=/tmp/sen2fire_local` el dataset se clona completo una
única vez y toda la sesión lee desde `/tmp`, que se borra al reiniciar la
máquina. Al terminar la clonación se escribe un sello `.dataset-cache.json` en
el destino: las corridas siguientes validan la copia local **sin tocar la
montura de red**, así que se puede entrenar con la NFS caída o desconectada.

## Uso

```bash
# smoke test del loader (carga dataset, imprime stats, prueba 1 batch)
python main.py

# entrenamiento mínimo (1 epoch, batch_size 1, 4 train + 2 val samples)
python train.py --epochs 1 --batch-size 1 --num-workers 0 --limit-train 4 --limit-val 2

# entrenamiento real
python train.py --epochs 10 --batch-size 2 --num-workers 2

# tests del flujo de carga (sin red, sin GPU, sin el dataset real)
pytest
```

## Estructura

```
src/sen2fire/
  data/
    config.py       lectura de .env por convención <PREFIX>_ROOT / <PREFIX>_LOCAL_CACHE
    source.py       DatasetSource: dónde vive el dataset y desde dónde se lee (NFS / caché)
    base.py         BaseSegmentationDataset: contrato común de los datasets
    registry.py     load_dataset(): nombre -> (config, fuente, clase)
    datamodule.py   SegmentationDataModule (LightningDataModule): prepara la fuente,
                    cachea el dataset y arma los DataLoaders train/val/test/predict
    datasets/
      sen2fire.py   Sen2FireNpzDataset: solo el formato .npz de Sen2Fire
  models/           U-Net, Dice + BCE
tests/
  conftest.py        fixtures (dataset sintético en tmp_path)
  test_data.py       tests del flujo de carga
  make_fake_data.py  genera un dataset sintético para probar sin la NFS
main.py             smoke test del loader
train.py            loop de entrenamiento
```

### Agregar un dataset

La fuente (dónde está y si se clona) y el formato (cómo se lee) son
independientes. Para soportar un dataset nuevo:

1. Subclasear `BaseSegmentationDataset` implementando `_index()` y
   `_load_sample()` en `src/sen2fire/data/datasets/<nombre>.py`.
2. `register("<nombre>", TuDataset)` en `registry.py`.
3. Agregar `<PREFIX>_ROOT` (y opcionalmente `<PREFIX>_LOCAL_CACHE`) al `.env`.

`main.py` y `train.py` no cambian. La clase recibe un `root` ya resuelto, así
que se puede testear contra un directorio temporal sin tocar el entorno:

```python
from sen2fire.data.datasets.sen2fire import Sen2FireNpzDataset

ds = Sen2FireNpzDataset(root="/tmp/fixtures/sen2fire", normalize="reflectance")
```