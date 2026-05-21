# Script 24.2 Testing V3.1: FILTRO DE CONTENCIÓN EN CUERPO DE VACA
# ==============================================================================
# LÓGICA: Adaptativo + Forma + FILTRO DE FONDO NEGRO (Intersección con la vaca)
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

def process_test_image(img_path, processor, output_dir):
    print(f"🧪 Procesando: {img_path.name}")
    img_cv = cv2.imread(str(img_path))
    if img_cv is None: return
    h_img, w_img = img_cv.shape[:2]
    
    # 1. Crear Máscara de la Vaca (Píxeles no negros)
    # Esto define dónde SÍ puede haber moscas
    gray_full = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, cow_mask = cv2.threshold(gray_full, 1, 255, cv2.THRESH_BINARY)
    # Pequeña erosión para no estar tan al borde
    cow_mask = cv2.erode(cow_mask, np.ones((3,3), np.uint8), iterations=1)

    # 2. Descubrimiento Avanzado (Detective)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray_full)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    combined = cv2.add(cv2.morphologyEx(gray_clahe, cv2.MORPH_BLACKHAT, kernel), 
                       cv2.morphologyEx(gray_clahe, cv2.MORPH_TOPHAT, kernel))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    _, thresh = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.dilate(thresh, np.ones((5,5), np.uint8), iterations=1)
    
    # IMPORTANTE: El detective solo busca dentro de la vaca
    thresh = cv2.bitwise_and(thresh, cow_mask)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 15 < area < 1500:
            x, y, w, h = cv2.boundingRect(cnt)
            box = [max(0, x-10)/w_img, max(0, y-10)/h_img, min(w_img, x+w+10)/w_img, min(h_img, y+h+10)/h_img]
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                candidates.append({'box': box, 'point': [int(M["m10"]/M["m00"])/w_img, int(M["m01"]/M["m00"])/h_img], 'cnt': cnt})

    # 3. Procesar con SAM 3
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
                    area = cv2.contourArea(c)
                    perim = cv2.arcLength(c, True)
                    circ = (4 * np.pi * area) / (perim * perim) if perim > 0 else 0
                    x, y, w, h = cv2.boundingRect(c)
                    aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 100
                    
                    # --- FILTRO DE CONTENCIÓN ---
                    # Comprobar si el centroide de la mosca está dentro de la máscara de la vaca
                    M_l = cv2.moments(c)
                    if M_l["m00"] != 0:
                        cx, cy = int(M_l["m10"]/M_l["m00"]), int(M_l["m01"]/M_l["m00"])
                        is_inside = cow_mask[cy, cx] > 0
                    else:
                        is_inside = False
                    
                    all_masks_info.append({'cnt': c, 'area': area, 'circ': circ, 'aspect': aspect, 'is_inside': is_inside})

    # 4. Estadísticas y Guardado
    if all_masks_info:
        areas = [m['area'] for m in all_masks_info if m['area'] > 5]
        median_area = np.median(areas)
        min_threshold = max(median_area * 0.4, 8)
        max_threshold = min(median_area * 3.0, 800)
    else:
        min_threshold, max_threshold = 20, 600

    img_limpio = img_cv.copy()
    valid_count = 0
    rejected_count = 0

    for m in all_masks_info:
        # AÑADIMOS 'is_inside' a la condición final
        if (min_threshold <= m['area'] <= max_threshold) and (m['circ'] > 0.25) and (m['aspect'] < 4.0) and m['is_inside']:
            cv2.drawContours(img_limpio, [m['cnt']], -1, (0, 255, 0), 2)
            valid_count += 1
        else:
            # Dibujar en blanco lo que está fuera de la vaca o descartado por forma
            cv2.drawContours(img_limpio, [m['cnt']], -1, (255, 255, 255), 1)
            rejected_count += 1

    # Añadir Azules (detective)
    for cand in candidates:
        cv2.drawContours(img_limpio, [cand['cnt']], -1, (255, 0, 0), 1)

    cv2.imwrite(str(output_dir / f"diag_CONTENCION_{img_path.name}"), img_limpio)
    print(f"✅ Finalizados: {valid_count} Verdes | {rejected_count} Blancos/Fuera")

def main():
    sam3_ckpt = "/data/estudiantes/vacas/sam3/checkpoints/sam3.pt"
    output_dir = Path("verificacion_etiquetas/test_v24_contencion")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar SAM 3
    device = torch.device("cuda:0")
    model = build_sam3_image_model(checkpoint_path=sam3_ckpt, device=str(device), load_from_HF=False)
    model.to(device)
    model.eval()
    processor = Sam3Processor(model, resolution=1008, device=str(device), confidence_threshold=0.15)

    # Imágenes de Test (La anterior y la nueva de referencia)
    test_images = [
        Path("runs/conteo_moscas_v24/roi_v24_2023-11-19_19.20.00-19.30.00[R][0@0][0]_f_0375_vaca0.jpg"),
        Path("runs/conteo_moscas_v24/roi_v24_2023-11-21_03.10.00-03.20.00[R][0@0][0]_f_0238_vaca2.jpg")
    ]

    for img_p in test_images:
        # Guardar copia original al lado
        cv2.imwrite(str(output_dir / f"ORIGINAL_{img_p.name}"), cv2.imread(str(img_p)))
        process_test_image(img_p, processor, output_dir)

if __name__ == "__main__":
    main()
