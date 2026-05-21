# Script 24.2 Testing V2: FILTRADO ADAPTATIVO POR ESCALA
# ==============================================================================

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# Configurar GPU 3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch

sys.path.append("/data/estudiantes/vacas/sam3")
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def main():
    img_path = Path("runs/conteo_moscas_v24/roi_v24_2023-11-19_19.20.00-19.30.00[R][0@0][0]_f_0375_vaca0.jpg")
    sam3_ckpt = "/data/estudiantes/vacas/sam3/checkpoints/sam3.pt"
    output_dir = Path("verificacion_etiquetas/test_v24_adaptativo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar SAM 3
    device = torch.device("cuda:0")
    model = build_sam3_image_model(checkpoint_path=sam3_ckpt, device=str(device), load_from_HF=False)
    model.to(device)
    model.eval()
    processor = Sam3Processor(model, resolution=1008, device=str(device), confidence_threshold=0.15)

    # 2. Descubrimiento Avanzado (Detective)
    img_cv = cv2.imread(str(img_path))
    h_img, w_img = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    combined = cv2.add(cv2.morphologyEx(gray_clahe, cv2.MORPH_BLACKHAT, kernel), 
                       cv2.morphologyEx(gray_clahe, cv2.MORPH_TOPHAT, kernel))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    _, thresh = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.dilate(thresh, np.ones((5,5), np.uint8), iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 15 < area < 2000: # Rango amplio para el detective
            x, y, w, h = cv2.boundingRect(cnt)
            box = [max(0, x-10)/w_img, max(0, y-10)/h_img, min(w_img, x+w+10)/w_img, min(h_img, y+h+10)/h_img]
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                candidates.append({'box': box, 'point': [int(M["m10"]/M["m00"])/w_img, int(M["m01"]/M["m00"])/h_img], 'cnt': cnt})

    # 3. Procesar con SAM 3 (Recolección de todas las máscaras posibles)
    pil_img = Image.open(str(img_path)).convert("RGB")
    state = processor.set_image(pil_img)
    processor.reset_all_prompts(state)
    for cand in candidates:
        state = processor.add_geometric_prompt(cand['box'], True, state)
        state = processor.add_geometric_prompt([cand['point'][0], cand['point'][1], 0.005, 0.005], True, state)
    
    all_masks_info = []
    if 'masks' in state and state['masks'] is not None:
        masks = state['masks']
        for i in range(masks.shape[0]):
            mask_np = (masks[i][0].cpu().numpy() * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if len(c) >= 3:
                    all_masks_info.append({'cnt': c, 'area': cv2.contourArea(c)})

    # --- LÓGICA DE FILTRADO ADAPTATIVO ---
    if all_masks_info:
        areas = [m['area'] for m in all_masks_info if m['area'] > 5] # Ignorar ruido extremo < 5px
        median_area = np.median(areas)
        # Definimos umbrales dinámicos (40% de la mediana como mínimo, 300% como máximo)
        min_threshold = median_area * 0.4
        max_threshold = median_area * 3.0
        
        # Seguridad: nunca menos de 8px ni más de 800px para evitar errores matemáticos
        min_threshold = max(min_threshold, 8)
        max_threshold = min(max_threshold, 800)
        
        print(f"📊 Estadísticas de la Imagen: Mediana={median_area:.1f}px | Umbral Dinámico=[{min_threshold:.1f} - {max_threshold:.1f}]")
    else:
        min_threshold, max_threshold, median_area = 20, 600, 0

    # --- GENERAR VERSIONES ---
    img_todo = img_cv.copy()
    img_limpio = img_cv.copy()
    
    valid_count = 0
    for m in all_masks_info:
        if min_threshold <= m['area'] <= max_threshold:
            # VERDE: Dentro del rango adaptativo
            cv2.drawContours(img_todo, [m['cnt']], -1, (0, 255, 0), 2)
            cv2.drawContours(img_limpio, [m['cnt']], -1, (0, 255, 0), 2)
            valid_count += 1
        else:
            # BLANCO: Fuera del rango adaptativo (Demasiado pequeño o grande para ESTA foto)
            cv2.drawContours(img_todo, [m['cnt']], -1, (255, 255, 255), 1)

    # Añadir azules (detective)
    for cand in candidates:
        cv2.drawContours(img_todo, [cand['cnt']], -1, (255, 0, 0), 2)
        cv2.drawContours(img_limpio, [cand['cnt']], -1, (255, 0, 0), 2)

    cv2.imwrite(str(output_dir / "diag_ADAPTATIVO_TODO.jpg"), img_todo)
    cv2.imwrite(str(output_dir / "diag_ADAPTATIVO_LIMPIO.jpg"), img_limpio)
    print(f"🚀 Resultados adaptativos guardados en: {output_dir}")
    print(f"✅ Etiquetas finales aceptadas: {valid_count}")

if __name__ == "__main__":
    main()
