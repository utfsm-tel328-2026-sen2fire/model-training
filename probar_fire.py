"""Prueba una U-Net entrenada con PyTorch Lightning sobre un parche de Sen2Fire.

Uso con rutas explícitas:
    uv run python probar_fire.py --ckpt checkpoints/.../best.ckpt \
        --npz /ruta/Sen2Fire/scene3/scene_3_patch_1_1.npz

Sin argumentos intenta abrir selectores de archivos (requiere tkinter y escritorio).

IMPORTANTE: importa la misma clase UNet que se utilizó al entrenar. La
normalización de entrada se realiza con Sen2FireDataset, NO sobre el NPZ crudo.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sen2fire.data.datasets.sen2fire import (
    Sen2FireNpzDataset,
    get_scene_split
)
from sen2fire.models import UNet


def elegir_archivo(ruta, titulo, patrones):
    """Usa la ruta de consola o abre un diálogo si no fue proporcionada."""
    if ruta is not None:
        archivo = Path(ruta).expanduser().resolve()
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog

            ventana = tk.Tk()
            ventana.withdraw()
            seleccionado = filedialog.askopenfilename(
                title=titulo, filetypes=[("Archivos compatibles", patrones), ("Todos", "*")]
            )
            ventana.destroy()
            if not seleccionado:
                raise SystemExit("No se seleccionó ningún archivo.")
            archivo = Path(seleccionado).resolve()
        except (ImportError, tk.TclError) as exc:
            raise SystemExit(
                f"No se pudo abrir el selector ({exc}). Usa --ckpt y --npz en la consola."
            ) from exc
    if not archivo.is_file():
        raise SystemExit(f"Archivo inexistente: {archivo}")
    return archivo


def cargar_unet_desde_ckpt(ruta_ckpt, device):
    """Carga un estado Lightning o un state_dict clásico en la misma U-Net."""
    # Solo abrir checkpoints propios o de una fuente confiable.
    checkpoint = torch.load(ruta_ckpt, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        raise RuntimeError("El checkpoint no contiene un diccionario de pesos.")
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    # Debe ser la MISMA arquitectura usada al entrenar.
    modelo = UNet(in_channels=12, n_classes=2)
    esperados = modelo.state_dict()
    primera_clave = next(iter(esperados))

    # Lightning suele agregar prefijos como 'model.', 'net.' o 'unet.'.
    prefijos = [clave[:-len(primera_clave)] for clave in state
                if clave.endswith(primera_clave)]

    for prefijo in prefijos:
        if all(prefijo + k in state and state[prefijo + k].shape == valor.shape
               for k, valor in esperados.items()):
            pesos_unet = {k: state[prefijo + k] for k in esperados}
            modelo.load_state_dict(pesos_unet, strict=True)
            modelo.to(device).eval()
            print(f"Pesos cargados correctamente (prefijo: {prefijo!r}).")
            return modelo

    print("Ejemplos de claves del checkpoint:", list(state)[:5])
    print("Ejemplos de claves que espera la U-Net:", list(esperados)[:5])
    raise RuntimeError(
        "Los pesos no coinciden con UNet(in_channels=12, out_channels=2). "
        "Revisa cómo se creó la U-Net dentro de tu LightningModule."
    )


def buscar_parche_en_dataset(ds, ruta_npz):
    """Recupera el parche por medio del dataset usado al entrenar."""
    # Los nombres de parche Sen2Fire contienen la escena y son únicos.
    coincidencias = [i for i, muestra in enumerate(ds.samples)
                     if ruta_npz.name in str(muestra)]
    if len(coincidencias) != 1:
        raise RuntimeError(
            f"Se encontraron {len(coincidencias)} coincidencias para {ruta_npz.name} "
            "en ds.samples. Revisa la ruta y la representación de ds.samples."
        )
    indice = coincidencias[0]
    image, label = ds[indice]
    return indice, image.float(), label.squeeze().long()


def crear_rgb(ruta_npz):
    """Prepara solo una visualización RGB; no modifica la entrada del modelo."""
    with np.load(ruta_npz, allow_pickle=False) as data:
        imagen = np.asarray(data["image"])
    if imagen.ndim != 3:
        raise ValueError(f"Imagen NPZ inesperada: {imagen.shape}")
    if imagen.shape[0] == 12:
        rgb = imagen[[3, 2, 1], :, :].transpose(1, 2, 0).astype(np.float32)
    elif imagen.shape[-1] == 12:
        rgb = imagen[:, :, [3, 2, 1]].astype(np.float32)
    else:
        raise ValueError(f"Se esperaban 12 canales, pero la forma es: {imagen.shape}")

    # Ajuste de contraste por canal para evitar que la imagen aparezca blanca.
    for canal in range(3):
        banda = rgb[:, :, canal]
        minimo, maximo = np.nanpercentile(banda, [2, 98])
        rgb[:, :, canal] = np.clip((banda - minimo) / (maximo - minimo + 1e-8), 0, 1)
    return np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)


def main():
    parser = argparse.ArgumentParser(description="Probar U-Net de fuego en Sen2Fire")
    parser.add_argument("--ckpt", type=Path, help="Checkpoint de Lightning")
    parser.add_argument("--npz", type=Path, help="Parche .npz de Sen2Fire")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Umbral de probabilidad para clasificar un píxel como fuego"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("predicciones"))
    parser.add_argument("--no-show", action="store_true", help="Guardar PNG sin abrir ventana")
    args = parser.parse_args()

    ckpt_path = elegir_archivo(args.ckpt, "Seleccionar checkpoint", "*.ckpt *.pt *.pth")
    npz_path = elegir_archivo(args.npz, "Seleccionar parche Sen2Fire", "*.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    modelo = cargar_unet_desde_ckpt(ckpt_path, device)

    # La normalización de Sen2FireDataset debe ser igual que en entrenamiento.
    ds = Sen2FireNpzDataset(
        root=npz_path.parent.parent,
        normalize="reflectance"
    )
    indice, imagen, etiqueta = buscar_parche_en_dataset(ds, npz_path)
    if imagen.shape[0] != 12:
        raise ValueError(f"El dataset devolvió {imagen.shape[0]} canales, no 12.")
    print(f"Parche cargado: {npz_path.name} (índice de dataset: {indice})")

    grupos = get_scene_split(ds)
    for grupo, indices in grupos.items():
        if indice in indices:
            print(f"Conjunto: {grupo}")
            break
    real = etiqueta.cpu().numpy()
    # [C,H,W] -> [1,C,H,W], para procesar un batch de una imagen.
    imagen_batch = imagen.unsqueeze(0).to(device)
    with torch.inference_mode():
        # Obtener las predicciones de la U-Net
        logits = modelo(imagen_batch)

            # Comprobar que existen dos clases
        if logits.shape[1] != 2:
            raise RuntimeError(
                f"Salida inesperada: {tuple(logits.shape)}"
            )

            # Convertir logits en probabilidades
        probabilities = torch.softmax(logits, dim=1)

            # Obtener únicamente la probabilidad de fuego
        prob_fuego = probabilities[0, 1].cpu().numpy()

            # Aplicar umbral de clasificación
        prediccion = (prob_fuego >= args.threshold).astype(np.uint8)

        print(f"Threshold: {args.threshold}")



    if prediccion.shape != real.shape:
        raise ValueError(f"Predicción {prediccion.shape} y etiqueta {real.shape} no coinciden")

    tp = int(((prediccion == 1) & (real == 1)).sum())
    fp = int(((prediccion == 1) & (real == 0)).sum())
    fn = int(((prediccion == 0) & (real == 1)).sum())

    print(f"\nThreshold: {args.threshold:.2f}")

    print(f"Fuego real: {tp + fn}")
    print(f"Fuego predicho: {tp + fp}")

    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")


    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else float("nan")

    print(f"Fuego real: {int((real == 1).sum())} píxeles")
    print(f"Fuego predicho: {int((prediccion == 1).sum())} píxeles")
    print(f"TP={tp}, FP={fp}, FN={fn}")
    print(f"Umbral utilizado: {args.threshold:.2f}")
    print(f"IoU={iou:.4f}, Dice={dice:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

    rgb = crear_rgb(npz_path)
    fig, ejes = plt.subplots(1, 3, figsize=(15, 5))
    nombres = ["Imagen RGB", "Máscara real (rojo = fuego)", "Predicción (rojo = fuego)"]
    for eje, nombre in zip(ejes, nombres):
        eje.imshow(rgb)
        eje.set_title(nombre)
        eje.axis("off")

    # Superponer solo píxeles con fuego; el fondo permanece transparente.
    for eje, mascara in [(ejes[1], real), (ejes[2], prediccion)]:
        overlay = np.zeros((*mascara.shape, 4), dtype=np.float32)
        overlay[mascara == 1] = [1.0, 0.0, 0.0, 0.65]  # rojo semitransparente
        eje.imshow(overlay)

    fig.suptitle(f"{npz_path.name} | Threshold={args.threshold:.2f} | IoU fuego = {iou:.4f} | Dice = {dice:.4f}")
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destino = args.output_dir / (
        f"{npz_path.stem}_threshold_{args.threshold:.2f}.png"
    )
    fig.savefig(destino, dpi=160, bbox_inches="tight")
    np.savez_compressed(
        args.output_dir / (
            f"{npz_path.stem}_threshold_{args.threshold:.2f}_prediccion.npz"
        ),
        label=prediccion.astype(np.uint8),
        prob_fire=prob_fuego
    )


    print(f"Comparación guardada en: {destino}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
