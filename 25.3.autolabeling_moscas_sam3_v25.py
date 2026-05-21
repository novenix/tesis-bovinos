# Script 25.3: Auto-Etiquetado con SAM 3 (Serie 25 - MODO SUPER-VIGILANTE V25 + ESCUDO)
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
    visual_dir = Path("verificacion_etiquetas/produccion_v25_sam3")
    visual_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 [GPU {gpu_id}] SAM 3 (CON ESCUDO V25) Cargando Modelo...")
    
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
                # Umbral más alto (15) para ignorar ruido de compresión JPEG en el fondo negro
                _, cow_mask = cv2.threshold(gray_orig, 15, 255, cv2.THRESH_BINARY)
                cow_mask = cv2.erode(cow_mask, np.ones((5,5), np.uint8), iterations=1)
                
                # --- 2. DESCUBRIMIENTO AVANZADO (MEJORADO ANTI-SOMBRAS) ---
                # CLAHE más suave para no exagerar sombras
                clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
                gray_clahe = clahe.apply(gray_orig)
                
                # Kernel más pequeño para moscas pequeñas
                kernel_mosca = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                
                # BlackHat (objetos oscuros sobre fondo claro) y TopHat (claros sobre oscuro)
                bh = cv2.morphologyEx(gray_clahe, cv2.MORPH_BLACKHAT, kernel_mosca)
                th = cv2.morphologyEx(gray_clahe, cv2.MORPH_TOPHAT, kernel_mosca)
                combined = cv2.add(bh, th)
                
                # Umbral Adaptativo en lugar de OTSU para manejar iluminación local
                thresh = cv2.adaptiveThreshold(combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
                
                # Filtro de Bordes (Laplaciano) para evitar sombras suaves
                laplacian = cv2.Laplacian(gray_clahe, cv2.CV_64F).var()
                
                # Aplicar escudo al detective
                thresh = cv2.bitwise_and(thresh, cow_mask)
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                candidates = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    # Bajamos el área máxima: una mosca no suele pasar de 400px en estos ROIs
                    if 10 < area < 400: 
                        # Validar "nitidez" local (evitar manchas difusas/sombras)
                        x, y, w, h = cv2.boundingRect(cnt)
                        roi_cnt = gray_clahe[y:y+h, x:x+w]
                        if roi_cnt.size > 0:
                            local_var = cv2.Laplacian(roi_cnt, cv2.CV_64F).var()
                            # Si la zona es muy suave (varianza baja), es probable que sea una sombra
                            if local_var < 100: continue
                            
                        box = [max(0, x-5)/w_img, max(0, y-5)/h_img, min(w_img, x+w+5)/w_img, min(h_img, y+h+5)/h_img]
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
                
                # Limitar cantidad de prompts para no saturar SAM y evitar falsos positivos
                # Priorizar los 100 candidatos más "prometedores" si hay demasiados
                if len(candidates) > 100:
                    candidates = sorted(candidates, key=lambda x: cv2.contourArea(x['cnt']), reverse=True)[:100]

                for cand in candidates:
                    state = processor.add_geometric_prompt(cand['box'], True, state)
                    state = processor.add_geometric_prompt(cand['point'] + [0.005, 0.005], True, state)
                
                if 'masks' in state and state['masks'] is not None:
                    masks = state['masks']
                    temp_masks_info = []
                    for i in range(masks.shape[0]):
                        mask_np = (masks[i][0].cpu().numpy() * 255).astype(np.uint8)
                        mask_np = cv2.bitwise_and(mask_np, cow_mask)
                        
                        m_cnts, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for c in m_cnts:
                            if len(c) >= 3:
                                a_l = cv2.contourArea(c)
                                p_l = cv2.arcLength(c, True)
                                circ = (4 * np.pi * a_l) / (p_l * p_l) if p_l > 0 else 0
                                x_b, y_b, w_b, h_b = cv2.boundingRect(c)
                                asp = max(w_b, h_b) / min(w_b, h_b) if min(w_b, h_b) > 0 else 100
                                
                                # Verificación de Centroide
                                M_l = cv2.moments(c)
                                if M_l["m00"] != 0:
                                    cx_l, cy_l = int(M_l["m10"]/M_l["m00"]), int(M_l["m01"]/M_l["m00"])
                                    is_inside = cow_mask[cy_l, cx_l] > 0 if cy_l < h_img and cx_l < w_img else False
                                else:
                                    is_inside = False
                                
                                temp_masks_info.append({'cnt': c, 'area': a_l, 'circ': circ, 'aspect': asp, 'is_inside': is_inside})
                    
                    # FILTRADO HÍBRIDO DEFINITIVO (MÁS ESTRICTO)
                    valid_polygons = []
                    rejected_for_viz = []
                    if temp_masks_info:
                        areas_list = [m['area'] for m in temp_masks_info if m['area'] > 5]
                        if areas_list:
                            median_area = np.median(areas_list)
                            # Moscas: tamaño consistente. Si algo es muy grande comparado con el resto, es sombra.
                            min_t = 8
                            max_t = 400 # Límite físico de una mosca real
                            for m in temp_masks_info:
                                # Subimos circularidad a 0.4 para evitar formas alargadas de sombras
                                if (min_t <= m['area'] <= max_t) and (m['circ'] > 0.4) and (m['aspect'] < 2.5) and m['is_inside']:
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
    # Escuchar la salida del script 25.2
    input_dir = Path("runs/conteo_moscas_v25")
    # Guardar en la nueva carpeta de etiquetas v25
    output_label_dir = Path("dataset_moscas/labels_v25")
    output_label_dir.mkdir(parents=True, exist_ok=True)

    manager = mp.Manager()
    task_queue = manager.Queue(maxsize=3000) 
    
    processes = []
    for gpu_id in AVAILABLE_GPUS:
        p = mp.Process(target=worker_process, args=(gpu_id, task_queue, output_label_dir, sam3_ckpt))
        p.start()
        processes.append(p)

    print(f"🕵️ SUPER-VIGILANTE V25 CON ESCUDO ACTIVADO. GPUs: {AVAILABLE_GPUS}")
    
    already_queued = set()
    while True:
        try:
            current_files = sorted(list(input_dir.glob("roi_v25_*.jpg")))
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
