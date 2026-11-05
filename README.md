# model-training

Entrenamiento de un U-Net para segmentación de fuego con imágenes proporcionadas por el satélite Sentinel-2.
El loop de entrenamiento corre sobre [PyTorch Lightning](https://lightning.ai/docs/pytorch) y se configura con un YAML por experimento.

## Setup

[uv](https://docs.astral.sh/uv/) gestiona el entorno: instala Python y las dependencias (incluido el paquete en modo editable).

Elige el backend de PyTorch según tu GPU:

| GPU              | Comando                  | Wheel que se instala                       |
| ---------------- | ------------------------ | ------------------------------------------ |
| NVIDIA           | `uv sync --extra cuda`   | `torch==2.11.0+cu128`  (último disponible) |
| AMD (ROCm ≥ 7.1) | `uv sync --extra rocm`   | `torch==2.13.0+rocm7.1` (último disponible) |

`uv.lock` incluye las dos variantes resueltas; se regenera con `uv lock --refresh` si cambian las dependencias o los índices de PyTorch.

`pytest` y `tensorboard` vienen en el grupo `dev`. Para agregar las herramientas de desarrollo sin volver a fijar el backend de torch:

```bash
uv sync --inexact    # no toca tu torch
```

## Variables de entorno (en `.env`)

| Variable | Requerida | Default | Descripción |
|---|---|---|---|
| `DATASET` | no | `sen2fire` | Qué dataset del registro cargar |
| `SEN2FIRE_ROOT` | sí | — | Ruta al dataset (NFS o local) con `scene1/`, `scene2/`, … |
| `SEN2FIRE_LOCAL_CACHE` | no | sin caché | Si se define, el dataset se clona una vez a esa ruta local |
| `NFS_SEN2FIRE_RUNS_ROOT` | no | `runs/` local | Raíz para checkpoints, TensorBoard y `summary.json` de cada run. Recomendado: una ruta NFS compartida entre colaboradores. |
| `USER` | no | `getpass.getuser()` | Usuario dentro de la ruta de runs (para que cada quien vea su subdirectorio). |

Las variables reales del entorno pisan a las del `.env`:

```bash
SEN2FIRE_LOCAL_CACHE= NFS_SEN2FIRE_RUNS_ROOT=/srv/nfs/sen2fire_runs python train.py fit --config configs/baseline.yaml
```

### Caché local del dataset

Con `SEN2FIRE_LOCAL_CACHE=/tmp/sen2fire_local` el dataset se clona completo una única vez y toda la sesión lee desde `/tmp`. Al terminar la clonación se escribe un sello `.dataset-cache.json` en el destino: las corridas siguientes validan la copia local **sin tocar la montura de red**, así que se puede entrenar con la NFS caída o desconectada.

## Entrenar

`train.py` es un *thin wrapper* sobre `LightningCLI`. Para correr el recipe del paper (5 niveles, base 64, BCEw + CE-weight fire=10, Adam + poly LR, 5000 iters, val cada 500):

```bash
# 1) Levantar deps con el backend que corresponda:
uv sync --inexact --extra cuda       # o --extra rocm

# 2) Revisar el YAML parseado antes de tirar la corrida larga:
python train.py fit --print-config --config configs/baseline.yaml | less

# 3) Correr:
python train.py fit --config configs/baseline.yaml
```

Cualquier flag del YAML se puede sobreescribir por CLI sin tocar el archivo:

```bash
python train.py fit --config configs/baseline.yaml trainer.max_steps=2000 model.base=32 data.batch_size=4
```

Sub-comandos soportados (los mismos que `LightningCLI`): `fit`, `validate`, `test`, `predict`.

### TensorBoard

`LightningCLI` arranca un `TensorBoardLogger` si el YAML provee uno (ver `configs/baseline.yaml` al final). Para ver las métricas en vivo:

```bash
tensorboard --logdir runs/<user>/<run_id>/tb_logs
# o si comparten NFS:
tensorboard --logdir $NFS_SEN2FIRE_RUNS_ROOT/<companion_user>
```

### Dónde caen los artefactos

Con `NFS_SEN2FIRE_RUNS_ROOT=/srv/nfs/sen2fire_runs` cada corrida escribe:

```
/srv/nfs/sen2fire_runs/<user>/<run_id>/
  configs/config.yaml     # snapshot del YAML usado (copia post-fit)
  checkpoints/best.ckpt   # mejor ckpt según val/iou_fire (ModelCheckpoint)
  tb_logs/events.*.tfevent
  stdout.log
  summary.json            # métricas finales + commit + duración
```

Sin la variable de entorno, todo cae a `./runs/<user>/<run_id>/` (gitignored).

### Registro de experimentos (sin git-lfs ni DVC)

Una línea JSON por run en `experiments.jsonl` (commiteado en git). Lo escribe el callback `ExperimentRegistryCallback` al final del fit, o al `on_exception` con `status="crashed"`.

Consultas rápidas con `jq`:

```bash
# últimas 5 corridas
tail -5 experiments.jsonl | jq '. | {user, created_at, status, best_val_iou_fire}'

# mejor run de un usuario
jq -c '. | select(.user == "khaen")' experiments.jsonl \
  | jq -s 'sort_by(-(.best_val_iou_fire // 0)) | .[0]'
```

### Reproducir un run

```bash
# Antes hay que hacer git checkout del commit del registro (con datamodule + yaml)
git checkout <commit> -- .                   # no toca tu working tree

# Si el checkpoint existe:
python -c "from sen2fire.training.reproduce import run; run('<run_id>')"

# O manualmente:
python train.py fit --config runs/<user>/<run_id>/configs/config.yaml --ckpt_path runs/<user>/<run_id>/checkpoints/best.ckpt
```

## Smoke test sin GPU ni red

```bash
pytest -q                          # 46 tests; CPU, sin red, sin dataset real
python main.py                     # smoke del loader (necesita SEN2FIRE_ROOT real)
```

`main.py` queda como un *sanity* del loader (no se reemplazó por `Lightning`).

## Estructura

```
src/sen2fire/
  data/
    config.py       lectura de .env por convención <PREFIX>_ROOT / <PREFIX>_LOCAL_CACHE
    source.py       DatasetSource: dónde vive el dataset y desde dónde se lee (NFS / caché)
    base.py         BaseSegmentationDataset: contrato común de los datasets
    registry.py     load_dataset(): nombre -> (config, fuente, clase)
    datamodule.py   SegmentationDataModule (pl.LightningDataModule): prepara la fuente,
                    cachea el dataset y arma los DataLoaders train/val/test/predict
    datasets/
      sen2fire.py   Sen2FireNpzDataset: solo el formato .npz de Sen2Fire
  models/           U-Net + (losses viven en training/losses.py)
  training/
    losses.py       CrossEntropyFireWeighted (paper: weight=[1, 10])
    optim.py        build_optimizer (adam|adamw|sgd) + build_scheduler (poly|cosine|none)
    lightning_module.py Sen2FireSegModule (pl.LightningModule) — el modelo de Lightning
    registry.py     ExperimentRegistryCallback + helpers de runs (NFS, JSONL)
    reproduce.py    list_all / find / best / run para experiments.jsonl
configs/
  baseline.yaml     recipe del paper Sen2Fire (defaults)
  experiments/      configs por colaborador / experimento
tests/
  conftest.py        fixtures (dataset sintético en tmp_path)
  test_data.py       carga (no GPU, no red)
  test_unet.py       arquitectura
  test_losses.py     losses
  test_lightning_module.py  fast_dev_run, optim, scheduler
  test_registry.py   callback de registro
  make_fake_data.py  util para escenarios sin NFS
experiments.jsonl    registro append-only (commiteado)
main.py              smoke del loader
train.py             LightningCLI entry (fit/validate/test/predict)
```

### Decisiones clave de implementación (migración al paper)

- **UNet**: 5 niveles, base 64, bilinear Upsample (`factor=2`). Replica exacto a `Baseline_code/model/Networks.py`. Defaults: `in_channels=12, n_classes=2`.
- **Loss**: `CrossEntropyLoss(weight=[1, weight_fire])` con `weight_fire=10`. Visto que el paper usa CE-only sobre las 2 clases + clase fire / non-fire; no usamos Dice ni BCE.
- **Optimizador**: Adam (`lr=1e-4`, `wd=5e-4`) por defecto; AdamW/SGD opcionales.
- **Schedule**: decaimiento polinomial `(1 - step/max_steps)**0.9` aplicado por iteración (`LambdaLR` con `interval='step'`). Cosine es alternativa.
- **Régimen**: step-based (`max_steps=5000`, `val_check_interval=500`).
- **Checkpoint**: `ModelCheckpoint(monitor="val/iou_fire", mode="max", save_top_k=1)` — guarda el mejor por IoU de la clase fire.
- **Métrica de progreso en `lightning_module.py`**: `val/iou_fire` (prog bar), `val/oa` (prog bar). Métricas secundarias loggeadas pero no en progress bar.
- **Registro de runs**: append-only JSONL en git. Checkpoints en NFS (configurable vía `NFS_SEN2FIRE_RUNS_ROOT`).
- **Reproducción**: `experiments.jsonl` lleva `git_commit` y el path al YAML snapshot bajo el NFS del run.

### Agregar un dataset

La fuente (dónde está y si se clona) y el formato (cómo se lee) son independientes. Para soportar un dataset nuevo:

1. Subclasear `BaseSegmentationDataset` implementando `_index()` y `_load_sample()` en `src/sen2fire/data/datasets/<nombre>.py`.
2. `register("<nombre>", TuDataset)` en `registry.py`.
3. Agregar `<PREFIX>_ROOT` (y opcionalmente `<PREFIX>_LOCAL_CACHE`) al `.env`.

`train.py`, `LightningModule` y los callbacks no cambian. La clase recibe un `root` ya resuelto, así que se puede testear contra un directorio temporal sin tocar el entorno.
