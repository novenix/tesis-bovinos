# Script 25.2: Extracción de ROI con Máscara e Inferencia SAHI (V25)
# ==============================================================================
# COMANDO DE EJECUCIÓN:
# nohup /opt/anaconda3/envs/tesis_vacas/bin/python -u 25.2.conteo_moscas_sahi_v25.py > logs/output_v25_sahi_mask.log 2>&1 &
# ==============================================================================

import os
# --- CONFIGURACIÓN DE AMBIENTE (DEBE IR ANTES DE CUALQUIER IMPORT DE TORCH/YOLO) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7" 
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def main():
    # --- MODELOS ---
    # Estadio 1: Usamos el modelo MEDIUM entrenado en el Script 17 para segmentar la vaca
    weights_st1 = "runs/segment/tesis_bovinos_medium/train_v17_medium/weights/best.pt"
    
    # --- RUTAS DE DATOS ---
    input_dirs = [
        Path("/data/estudiantes/vacas/dataset_moscas_v2/random_images/finalDatasetMoscas")
    ]
    output_dir = Path("runs/conteo_moscas_v25")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🕒 Cargando modelo de segmentación de bovinos...")
    model_st1 = YOLO(weights_st1)
    print("✅ Modelo listo.")

    all_images = []
    for d in input_dirs:
        if d.exists():
            all_images.extend(list(d.glob("*.jpg")) + list(d.glob("*.png")))
            
    total_imgs = len(all_images)
    print(f"🚀 Extrayendo ROIs de {total_imgs} imágenes.")

    processed_count = 0
    for img_p in all_images:
        img = cv2.imread(str(img_p))
        if img is None: continue
        
        # --- ESTADIO 1: SEGMENTACIÓN DE VACA ---
        results_st1 = model_st1(img, verbose=False, device="cuda:0")[0]
        
        if results_st1.masks is not None:
            for i, mask in enumerate(results_st1.masks.data):
                cls_id = int(results_st1.boxes.cls[i])
                if cls_id != 0: continue 

                # 1. Crear máscara binaria
                mask_np = mask.cpu().numpy()
                mask_rescaled = cv2.resize(mask_np, (img.shape[1], img.shape[0]))
                mask_binary = (mask_rescaled > 0.5).astype(np.uint8) * 255
                
                # 2. BITWISE AND: Borrar el fondo (Limpieza nuclear)
                masked_img = cv2.bitwise_and(img, img, mask=mask_binary)
                
                # 3. Recortar ROI
                box = results_st1.boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box
                # Añadir un pequeño margen de 10px para no cortar moscas en el borde
                h_img, w_img = img.shape[:2]
                y1, y2 = max(0, y1-10), min(h_img, y2+10)
                x1, x2 = max(0, x1-10), min(w_img, x2+10)
                
                roi_masked = masked_img[y1:y2, x1:x2].copy()
                
                # Guardar el ROI enmascarado
                roi_name = f"roi_v25_{img_p.stem}_vaca{i}.jpg"
                cv2.imwrite(str(output_dir / roi_name), roi_masked)
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"📊 Extracción: {processed_count}/{total_imgs} ({(processed_count/total_imgs)*100:.1f}%)")
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"📊 Progreso: {processed_count}/{total_imgs} ({(processed_count/total_imgs)*100:.1f}%)")

if __name__ == "__main__":
    main()
