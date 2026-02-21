import os
import sys
import argparse

# --- CONFIGURACIÓN DE GPU ANTES DE IMPORTAR TORCH ---
def pre_parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=int)
    args, _ = parser.parse_known_args()
    return args

args_pre = pre_parse_args()
if args_pre.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_pre.gpu)
# ----------------------------------------------------

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import time
import pandas as pd

# Añadir el repositorio clonado al path
sys.path.append("/data/estudiantes/vacas/sam3")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def parse_args():
    parser = argparse.ArgumentParser(description="Segmentación Masiva con SAM 3 Multi-GPU")
    parser.add_argument("--gpu", type=int, required=True, help="ID de la GPU a usar (1, 2, 3 o 4)")
    parser.add_argument("--shard", type=int, required=True, help="Índice del fragmento (0, 1, 2 o 3)")
    parser.add_argument("--total_shards", type=int, default=4, help="Número total de fragmentos")
    return parser.parse_args()

def save_yolo_bbox(path, class_id, box, img_w, img_h):
    x1, y1, x2, y2 = box
    dw, dh = 1.0/img_w, 1.0/img_h
    x, y = (x1 + x2)/2.0 * dw, (y1 + y2)/2.0 * dh
    w, h = (x2 - x1) * dw, (y2 - y1) * dh
    with open(path, 'a') as f:
        f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def save_yolo_seg(path, class_id, mask, img_w, img_h):
    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    with open(path, 'a') as f:
        for contour in contours:
            if len(contour) < 3: continue
            line = f"{class_id} " + " ".join([f"{p[0][0]/img_w:.6f} {p[0][1]/img_h:.6f}" for p in contour])
            f.write(line + "\n")

def draw_result(img, box, mask, color, label):
    mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, 1)
    overlay = img.copy()
    cv2.fillPoly(overlay, contours, color)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    cv2.putText(img, label, (int(box[0]), int(box[1])-5), 0, 0.5, color, 2)

def main():
    args = parse_args()
    # No configuramos environ aquí porque ya lo hicimos arriba
    
    # Configuración de Rutas
    image_dir = Path("/data/estudiantes/vacas/dataset_curado/train/images")
    label_root = Path("/data/estudiantes/vacas/dataset_curado/train/labels")
    qc_dir = Path("/data/estudiantes/vacas/verificacion_etiquetas/produccion_sam3")
    log_file = Path(f"logs/progreso_sam3_gpu{args.gpu}.csv")
    
    dirs = {
        "fc_bbox": label_root / "full_cow/bbox", "fc_seg": label_root / "full_cow/seg",
        "kp_bbox": label_root / "key_parts/bbox", "kp_seg": label_root / "key_parts/seg"
    }
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Cargar Registro de Reanudación
    processed_images = set()
    if log_file.exists():
        try:
            df = pd.read_csv(log_file)
            if 'filename' in df.columns:
                processed_images = set(df['filename'].tolist())
        except:
            pass

    # Cargar SAM 3
    print(f"🚀 [GPU {args.gpu}] Cargando SAM 3...")
    # Forzamos a que use 'cuda:0' porque para el proceso la GPU asignada es la ÚNICA que existe (gracias a CUDA_VISIBLE_DEVICES)
    model = build_sam3_image_model(checkpoint_path="/data/estudiantes/vacas/sam3/checkpoints/sam3.pt", device="cuda", load_from_HF=False)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=0.3)

    # Sharding
    all_images = sorted(list(image_dir.glob("*.jpg")))
    shard_size = len(all_images) // args.total_shards
    my_shard_images = all_images[args.shard * shard_size : (args.shard + 1) * shard_size] if args.shard < args.total_shards - 1 else all_images[args.shard * shard_size:]
    
    # Solo procesar las que faltan
    my_images = [f for f in my_shard_images if f.name not in processed_images]
    
    print(f"📦 [GPU {args.gpu}] Fragmento {args.shard}: {len(my_images)} imágenes pendientes de un total de {len(my_shard_images)}.")

    class_map = {"cow": 0, "head": 1, "leg": 2, "tail": 3}
    
    count = len(processed_images)
    for img_path in my_images:
        try:
            img_pil = Image.open(img_path).convert("RGB")
            w, h = img_pil.size
            do_qc = (count % 500 == 0)
            img_qc_fc = cv2.imread(str(img_path)) if do_qc else None
            img_qc_kp = cv2.imread(str(img_path)) if do_qc else None

            # Inferencia Vaca
            state = processor.set_image(img_pil)
            state = processor.set_text_prompt("cow", state)
            if "boxes" in state and len(state["boxes"]) > 0:
                for i in range(len(state["boxes"])):
                    box, score, mask = state["boxes"][i].cpu().numpy(), state["scores"][i].cpu().numpy(), state["masks"][i, 0]
                    save_yolo_bbox(dirs["fc_bbox"] / f"{img_path.stem}.txt", 0, box, w, h)
                    save_yolo_seg(dirs["fc_seg"] / f"{img_path.stem}.txt", 0, mask, w, h)
                    if do_qc: draw_result(img_qc_fc, box, mask, (0, 255, 0), f"cow {score:.2f}")

            # Inferencia Partes
            for part in ["head", "leg", "tail"]:
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(f"cow {part}", state)
                if "boxes" in state and len(state["boxes"]) > 0:
                    color = (255, 0, 0) if part == "head" else (0, 165, 255) if part == "leg" else (255, 255, 0)
                    for i in range(len(state["boxes"])):
                        box, mask = state["boxes"][i].cpu().numpy(), state["masks"][i, 0]
                        save_yolo_bbox(dirs["kp_bbox"] / f"{img_path.stem}.txt", class_map[part], box, w, h)
                        save_yolo_seg(dirs["kp_seg"] / f"{img_path.stem}.txt", class_map[part], mask, w, h)
                        if do_qc: draw_result(img_qc_kp, box, mask, color, part)

            # Guardar QC si aplica
            if do_qc:
                cv2.imwrite(str(qc_dir / f"qc_gpu{args.gpu}_fc_{img_path.name}"), img_qc_fc)
                cv2.imwrite(str(qc_dir / f"qc_gpu{args.gpu}_kp_{img_path.name}"), img_qc_kp)

            # Registrar progreso
            header = not log_file.exists()
            with open(log_file, "a") as f:
                if header: f.write("filename\n")
                f.write(f"{img_path.name}\n")
            
            count += 1
            if count % 100 == 0: 
                print(f"  [GPU {args.gpu}] {count}/{len(my_shard_images)} procesadas...")

        except Exception as e:
            print(f"❌ Error en {img_path.name}: {e}")

    print(f"✅ [GPU {args.gpu}] Fragmento completado.")

if __name__ == "__main__":
    main()
