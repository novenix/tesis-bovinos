# Script 24.1: Extracción de ROI con Máscara e Inferencia SAHI (V4)
# ==============================================================================
# COMANDO DE EJECUCIÓN:
# nohup /opt/anaconda3/envs/tesis_vacas/bin/python -u 24.1.conteo_moscas_sahi_v4.py > logs/output_v24.1_sahi_mask.log 2>&1 &
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
    # Estadio 2: Modelo de moscas para SAHI
    weights_st2 = "yolo26m-seg.pt" 
    
    # --- RUTAS DE DATOS ---
    input_dirs = [
        Path("dataset_final_yolo/train/images"),
        Path("dataset_final_yolo/val/images")
    ]
    output_dir = Path("runs/conteo_moscas_v24")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🕒 Cargando modelos en GPU...")
    model_st1 = YOLO(weights_st1)
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=weights_st2,
        confidence_threshold=0.50,
        device="cuda:0" # Corresponde a la física 2
    )
    print("✅ Modelos listos.")

    all_images = []
    for d in input_dirs:
        if d.exists():
            all_images.extend(list(d.glob("*.jpg")) + list(d.glob("*.png")))
            
    total_imgs = len(all_images)
    print(f"🚀 Procesando {total_imgs} imágenes totales.")

    processed_count = 0
    for img_p in all_images:
        # Verificar si ya existe el ROI para no repetir (opcional, por si se cae el script)
        # roi_name = f"roi_v24_{img_p.stem}_vaca0.jpg"
        # if (output_dir / roi_name).exists(): 
        #     processed_count += 1
        #     continue

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
                
                # 2. BITWISE AND: Borrar el fondo
                masked_img = cv2.bitwise_and(img, img, mask=mask_binary)
                
                # 3. Recortar ROI
                box = results_st1.boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box
                roi_masked = masked_img[y1:y2, x1:x2].copy()
                
                # Guardar el ROI enmascarado para SAM 3
                roi_name = f"roi_v24_{img_p.stem}_vaca{i}.jpg"
                cv2.imwrite(str(output_dir / roi_name), roi_masked)

                # --- ESTADIO 2: SAHI (Moscas) ---
                result_sahi = get_sliced_prediction(
                    roi_masked,
                    detection_model,
                    slice_height=256,
                    slice_width=256,
                    overlap_height_ratio=0.1,
                    overlap_width_ratio=0.1,
                    verbose=0
                )
                
                # Guardar visualización si hay moscas
                if len(result_sahi.object_prediction_list) > 0:
                    res_name = f"res_v24_{img_p.stem}_vaca{i}.jpg"
                    result_sahi.export_visuals(export_dir=str(output_dir), file_name=res_name.replace(".jpg", ""))
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"📊 Progreso: {processed_count}/{total_imgs} ({(processed_count/total_imgs)*100:.1f}%)")

if __name__ == "__main__":
    main()
