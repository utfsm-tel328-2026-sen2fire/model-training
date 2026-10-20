import torch
import numpy as np
import matplotlib.pyplot as plt
from sen2fire.models import UNet

def mask_ensamble():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=12, out_channels=3).to(device)

    # Cargar pesos del modelo desde un checkpoint
    weight_path = "/home/antonio/TK_Stratego/Shared/Dataset/Pesos/Segundo_Peso/best.pt"
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Cargamos imagen del stratego -> dataset
    npz_image_path = "/home/antonio/TK_Stratego/Shared/Dataset/Sen2Fire/scene3/scene_3_patch_14_23.npz"
    data = np.load(npz_image_path)
    img_np = data['image'].astype(np.float32)

    img_np = np.clip(img_np/ 10000  , 0, 1)  # Normalizamos a [0, 1]
    if img_np.shape[-1] == 12:
        img_np = np.transpose(img_np, (2, 0, 1))

    # Covierte a tensor de PyTorch y añade dimensión de batch
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        predict = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    rgb_mask = np.zeros((512, 512, 3), dtype=np.uint8)

    # Asignar colores a cada clase
    bg_color = [0, 0, 0] # Fondo: negro
    scar_color = [139, 69, 19] # Cicatriz: Cafe
    fire_color = [255, 0, 0] # Fuego: Rojo

    rgb_mask[predict == 0] = bg_color
    rgb_mask[predict == 1] = scar_color
    rgb_mask[predict == 2] = fire_color

    # Construccion de colores para los otros canales
    false_color = np.zeros((512,512,3), dtype=np.float32)
    false_color[:, :, 0] = img_np[11, :,:] * 2.5
    false_color[:, :, 1] = img_np[8, :,:] * 2.5
    false_color[:, :, 2] = img_np[2, :,:] * 2.5
    false_color = np.clip(false_color, 0.0, 1.0)

    fig, (ax1, ax2) = plt.subplots(1 ,2 , figsize=(15,7))

    ax1.imshow(false_color)
    ax1.set_title("Imagen Original con color reconstruido")
    ax1.axis("off")

    ax2.imshow(rgb_mask)
    ax2.set_title("M3 POST-MASK: Predicción del modelo", fontsize=14)
    ax2.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    mask_ensamble()
