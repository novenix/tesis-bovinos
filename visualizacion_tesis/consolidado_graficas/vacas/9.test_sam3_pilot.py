import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import time

# Añadir el directorio del repositorio clonado al path
sys.path.append("/data/estudiantes/vacas/sam3")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def save_yolo_bbox(path, class_id, box, img_w, img_h):
    """Guarda en formato YOLO detección: class x_center y_center width height (normalizado)"""
    x1, y1, x2, y2 = box
    dw = 1.0 / img_w
    dh = 1.0 / img_h
    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    with open(path, 'a') as f:
        f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def save_yolo_seg(path, class_id, mask, img_w, img_h):
    """Guarda en formato YOLO segmentación: class x1 y1 x2 y2 ... (normalizado)"""
    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    with open(path, 'a') as f:
        for contour in contours:
            if len(contour) < 3: continue 
            flat_contour = contour.flatten()
            line = f"{class_id}"
            for i in range(0, len(flat_contour), 2):
                x = flat_contour[i] / img_w
                y = flat_contour[i+1] / img_h
                line += f" {x:.6f} {y:.6f}"
            f.write(line + "\n")

def draw_result(img, box, mask, color, label):
    """Dibuja polígono y box sobre la imagen"""
    # Polígono
    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, 1)
    overlay = img.copy()
    cv2.fillPoly(overlay, contours, color)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    # BBox
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    cv2.putText(img, label, (int(box[0]), int(box[1])-5), 0, 0.5, color, 2)

def main():
    # Rutas
    checkpoint = "/data/estudiantes/vacas/sam3/checkpoints/sam3.pt"
    image_dir = Path("/data/estudiantes/vacas/dataset_curado/train/images")
    label_root = Path("/data/estudiantes/vacas/dataset_curado/train/labels_pilot")
    vis_root = Path("/data/estudiantes/vacas/verificacion_etiquetas/piloto_sam3")
    
    # Crear Estructura de Carpetas
    dirs = {
        "fc_bbox": label_root / "full_cow/bbox",
        "fc_seg": label_root / "full_cow/seg",
        "kp_bbox": label_root / "key_parts/bbox",
        "kp_seg": label_root / "key_parts/seg",
        "vis_fc": vis_root / "full_cow",
        "vis_kp": vis_root / "key_parts"
    }
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)

    # Cargar SAM 3
    print("🚀 Cargando SAM 3...")
    model = build_sam3_image_model(checkpoint_path=checkpoint, device="cuda", load_from_HF=False)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=0.3)

    # Imágenes de prueba (las mismas 10)
    all_images = sorted(list(image_dir.glob("*.jpg")))
    image_paths = all_images[::len(all_images)//10][:10]
    
    CLASS_MAP = {"cow": 0, "head": 1, "leg": 2, "tail": 3}
    
    for img_path in image_paths:
        print(f"\n📸 Procesando: {img_path.name}")
        img_pil = Image.open(img_path).convert("RGB")
        w, h = img_pil.size
        img_vis_fc = cv2.imread(str(img_path))
        img_vis_kp = cv2.imread(str(img_path))
        
        # 1. FULL COW
        state = processor.set_image(img_pil)
        state = processor.set_text_prompt("cow", state)
        if "boxes" in state and len(state["boxes"]) > 0:
            for i in range(len(state["boxes"])):
                box, score, mask = state["boxes"][i].cpu().numpy(), state["scores"][i].cpu().numpy(), state["masks"][i, 0]
                save_yolo_bbox(dirs["fc_bbox"] / f"{img_path.stem}.txt", 0, box, w, h)
                save_yolo_seg(dirs["fc_seg"] / f"{img_path.stem}.txt", 0, mask, w, h)
                draw_result(img_vis_fc, box, mask, (0, 255, 0), f"cow {score:.2f}")

        # 2. KEY PARTS
        for part in ["head", "leg", "tail"]:
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(f"cow {part}", state)
            if "boxes" in state and len(state["boxes"]) > 0:
                color = (255, 0, 0) if part == "head" else (0, 165, 255) if part == "leg" else (255, 255, 0)
                for i in range(len(state["boxes"])):
                    box, mask = state["boxes"][i].cpu().numpy(), state["masks"][i, 0]
                    save_yolo_bbox(dirs["kp_bbox"] / f"{img_path.stem}.txt", CLASS_MAP[part], box, w, h)
                    save_yolo_seg(dirs["kp_seg"] / f"{img_path.stem}.txt", CLASS_MAP[part], mask, w, h)
                    draw_result(img_vis_kp, box, mask, color, part)

        # Guardar Visualizaciones
        cv2.imwrite(str(dirs["vis_fc"] / f"fc_{img_path.name}"), img_vis_fc)
        cv2.imwrite(str(dirs["vis_kp"] / f"kp_{img_path.name}"), img_vis_kp)

    print(f"\n✅ Piloto Finalizado. Carpetas creadas en {label_root} y {vis_root}")

if __name__ == "__main__":
    main()
