# Script 24.2: Auto-Etiquetado con SAM 3 (Serie 24 - MODO SUPER-VIGILANTE V2 + ESCUDO)
# ==============================================================================
# LÓGICA: Descubrimiento Dual + Box Prompts + Filtro Adaptativo Mediana + Filtro Forma
# MEJORA: ESCUDO DE CONTENCIÓN (Solo etiquetas dentro del cuerpo de la vaca)
# ==============================================================================

import os
import sys
import cv2
import time
import numpy as np
import torch.multiprocessing as mp
from pathlib import Path
from PIL import Image

# GPUs configuradas (3 a 7)
AVAILABLE_GPUS = [3, 4, 5, 6, 7]

def worker_process(gpu_id, task_queue, output_label_dir, sam3_ckpt):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    import torch
    sys.path.append("/data/estudiantes/vacas/sam3")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = torch.device("cuda:0")
    visual_dir = Path("verificacion_etiquetas/produccion_v24_sam3")
    visual_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 [GPU {gpu_id}] SAM 3 (CON ESCUDO) Cargando Modelo...")
    
    try:
        model = build_sam3_image_model(checkpoint_path=sam3_ckpt, device=str(device), load_from_HF=False)
        model.to(device)
        model.eval()
        processor = Sam3Processor(model, resolution=1008, device=str(device), confidence_threshold=0.15)
        
        processed_count = 0
        while True:
            img_p = task_queue.get()
            if img_p is None: break 
                
            out_label = output_label_dir / f"{img_p.stem}.txt"
            if out_label.exists() and os.path.getsize(out_label) > 0: continue

            try:
                img_cv = cv2.imread(str(img_p))
                if img_cv is None: continue
                h_img, w_img = img_cv.shape[:2]
                
                # --- 1. ESCUDO DE CONTENCIÓN (Cuerpo de la vaca) ---
                gray_orig = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                _, cow_mask = cv2.threshold(gray_orig, 1, 255, cv2.THRESH_BINARY)
                cow_mask = cv2.erode(cow_mask, np.ones((3,3), np.uint8), iterations=1)
                
                # --- 2. DESCUBRIMIENTO AVANZADO ---
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray_clahe = clahe.apply(gray_orig)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
                combined = cv2.add(cv2.morphologyEx(gray_clahe, cv2.MORPH_BLACKHAT, kernel), 
                                   cv2.morphologyEx(gray_clahe, cv2.MORPH_TOPHAT, kernel))
                combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
                _, thresh = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh = cv2.dilate(thresh, np.ones((5,5), np.uint8), iterations=1)
                
                # Aplicar escudo al detective
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
                            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                            candidates.append({'box': box, 'point': [cx/w_img, cy/h_img], 'cnt': cnt})
                
                if not candidates: 
                    open(out_label, 'w').close()
                    continue

                # --- 3. SAM 3 ---
                pil_img = Image.open(str(img_p)).convert("RGB")
                state = processor.set_image(pil_img)
                processor.reset_all_prompts(state)
                for cand in candidates:
                    state = processor.add_geometric_prompt(cand['box'], True, state)
                    state = processor.add_geometric_prompt(cand['point'] + [0.005, 0.005], True, state)
                
                if 'masks' in state and state['masks'] is not None:
                    masks = state['masks']
                    temp_masks_info = []
                    for i in range(masks.shape[0]):
                        mask_np = (masks[i][0].cpu().numpy() * 255).astype(np.uint8)
                        m_cnts, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for c in m_cnts:
                            if len(c) >= 3:
                                a_l = cv2.contourArea(c)
                                p_l = cv2.arcLength(c, True)
                                circ = (4 * np.pi * a_l) / (p_l * p_l) if p_l > 0 else 0
                                x_b, y_b, w_b, h_b = cv2.boundingRect(c)
                                asp = max(w_b, h_b) / min(w_b, h_b) if min(w_b, h_b) > 0 else 100
                                
                                # Verificación de Escudo (Centroide dentro de la vaca)
                                M_l = cv2.moments(c)
                                if M_l["m00"] != 0:
                                    cx_l, cy_l = int(M_l["m10"]/M_l["m00"]), int(M_l["m01"]/M_l["m00"])
                                    is_inside = cow_mask[cy_l, cx_l] > 0 if cy_l < h_img and cx_l < w_img else False
                                else:
                                    is_inside = False
                                
                                temp_masks_info.append({'cnt': c, 'area': a_l, 'circ': circ, 'aspect': asp, 'is_inside': is_inside})
                    
                    # FILTRADO HÍBRIDO DEFINITIVO
                    valid_polygons = []
                    rejected_for_viz = []
                    if temp_masks_info:
                        areas_list = [m['area'] for m in temp_masks_info if m['area'] > 5]
                        if areas_list:
                            median_area = np.median(areas_list)
                            min_t = max(median_area * 0.4, 8)
                            max_t = min(median_area * 3.0, 800)
                            for m in temp_masks_info:
                                if (min_t <= m['area'] <= max_t) and (m['circ'] > 0.25) and (m['aspect'] < 4.0) and m['is_inside']:
                                    poly = [f"{pt[0][0]/w_img:.6f} {pt[0][1]/h_img:.6f}" for pt in m['cnt']]
                                    valid_polygons.append(f"0 {' '.join(poly)}\n")
                                else:
                                    rejected_for_viz.append(m['cnt'])
                    
                    if valid_polygons:
                        with open(out_label, 'w') as f:
                            f.writelines(valid_polygons)
                        processed_count += 1
                        if processed_count % 50 == 0:
                            draw_img = img_cv.copy()
                            for cand in candidates: cv2.drawContours(draw_img, [cand['cnt']], -1, (255, 0, 0), 2)
                            for r_cnt in rejected_for_viz: cv2.drawContours(draw_img, [r_cnt], -1, (255, 255, 255), 1)
                            for poly_line in valid_polygons:
                                pts = np.array([float(x) for x in poly_line.strip().split()[1:]]).reshape(-1, 2)
                                pts[:, 0] *= w_img; pts[:, 1] *= h_img
                                cv2.polylines(draw_img, [pts.astype(np.int32)], True, (0, 255, 0), 2)
                            cv2.imwrite(str(visual_dir / f"check_SUPER_{img_p.name}"), draw_img)
                    else:
                        open(out_label, 'w').close()

                # Limpiar memoria
                del state
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"⚠️ Error en {img_p.name}: {e}")
                torch.cuda.empty_cache()

    except Exception as e:
        print(f"🔥 Error fatal: {e}")

def main():
    mp.set_start_method('spawn', force=True)
    sam3_ckpt = "/data/estudiantes/vacas/sam3/checkpoints/sam3.pt"
    input_dir = Path("runs/conteo_moscas_v24")
    output_label_dir = Path("dataset_moscas/labels_v24")
    output_label_dir.mkdir(parents=True, exist_ok=True)

    manager = mp.Manager()
    task_queue = manager.Queue(maxsize=3000) 
    
    processes = []
    for gpu_id in AVAILABLE_GPUS:
        p = mp.Process(target=worker_process, args=(gpu_id, task_queue, output_label_dir, sam3_ckpt))
        p.start()
        processes.append(p)

    print(f"🕵️ SUPER-VIGILANTE V2 CON ESCUDO ACTIVADO. GPUs: {AVAILABLE_GPUS}")
    
    already_queued = set()
    while True:
        try:
            current_files = sorted(list(input_dir.glob("roi_v24_*.jpg")))
            new_files_count = 0
            for f in current_files:
                label_f = output_label_dir / f"{f.stem}.txt"
                if f.name not in already_queued and not label_f.exists():
                    task_queue.put(f)
                    already_queued.add(f.name)
                    new_files_count += 1
            if new_files_count > 0:
                print(f"➕ Añadidas {new_files_count} nuevas imágenes a la cola.")
            time.sleep(120)
            if len(already_queued) > 150000: already_queued.clear()
        except KeyboardInterrupt: break
        except Exception: time.sleep(10)

if __name__ == "__main__":
    main()
