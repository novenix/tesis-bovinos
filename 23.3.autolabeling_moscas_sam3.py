# Script 23.3 (Optimizado): Auto-Etiquetado de Moscas con SAM 3
# ==============================================================================
# COMANDO DE EJECUCIÓN (Copiar y pegar):
# nohup /opt/anaconda3/envs/tesis_vacas/bin/python -u 23.3.autolabeling_moscas_sam3.py > logs/output_v23.3_autolabeling.log 2>&1 &
# ==============================================================================

import os
import sys
import cv2
import numpy as np
import torch.multiprocessing as mp
from pathlib import Path
from PIL import Image

# --- CONFIGURACIÓN DE AMBIENTE ---
AVAILABLE_GPUS = [2, 3, 4, 5, 6, 7]

def worker_process(gpu_id, task_queue, output_label_dir, sam3_ckpt):
    """
    Proceso Consumidor: Extrae imágenes de la cola dinámicamente.
    Aislamiento estricto: Se realiza ANTES de importar torch.
    """
    # AISLAMIENTO CRÍTICO
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Importaciones pesadas DENTRO del proceso para respetar el aislamiento
    import torch
    sys.path.append("/data/estudiantes/vacas/sam3")
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError:
        print(f"❌ [GPU {gpu_id}] Error: No se encuentra SAM 3.")
        return

    device = torch.device("cuda:0") # Al aislar, cada proceso ve su GPU como cuda:0
    visual_dir = Path("verificacion_etiquetas/produccion_sam3")
    visual_dir.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    print(f"🚀 [GPU {gpu_id}] Inicializando SAM 3 en {device}...")
    
    try:
        model = build_sam3_image_model(checkpoint_path=sam3_ckpt, device=str(device), load_from_HF=False)
        model.to(device)
        model.eval()
        processor = Sam3Processor(model, resolution=1008, device=str(device), confidence_threshold=0.3)
        
        while True:
            img_p = task_queue.get()
            if img_p is None: break
                
            out_label = output_label_dir / f"{img_p.stem}.txt"
            if out_label.exists() and os.path.getsize(out_label) > 0:
                continue

            try:
                img_cv = cv2.imread(str(img_p))
                if img_cv is None: continue
                
                # FASE A: DISCOVERY (Black-Hat)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
                _, thresh = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                points = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 5 < area < 150: 
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            points.append([int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])])
                
                if not points: continue

                # FASE B: REFINEMENT (SAM 3)
                pil_img = Image.open(str(img_p)).convert("RGB")
                w, h = pil_img.size
                state = processor.set_image(pil_img)
                processor.reset_all_prompts(state)
                
                for pt in points:
                    state = processor.add_geometric_prompt([pt[0]/w, pt[1]/h, 0.005, 0.005], True, state)
                
                if 'masks' in state and state['masks'] is not None:
                    masks = state['masks']
                    valid_polygons = []
                    draw_img = img_cv.copy() if processed_count % 100 == 0 else None
                    
                    for i in range(masks.shape[0]):
                        mask_np = (masks[i][0].cpu().numpy() * 255).astype(np.uint8)
                        cnts, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for c in cnts:
                            if len(c) >= 3:
                                poly = [f"{pt[0][0]/w:.6f} {pt[0][1]/h:.6f}" for pt in c]
                                valid_polygons.append(f"0 {' '.join(poly)}\n")
                                if draw_img is not None:
                                    cv2.drawContours(draw_img, [c], -1, (0, 255, 0), 1)
                    
                    if valid_polygons:
                        with open(out_label, 'w') as f:
                            f.writelines(valid_polygons)
                        if draw_img is not None:
                            cv2.imwrite(str(visual_dir / f"sample_{gpu_id}_{img_p.name}"), draw_img)
                
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"📊 [GPU {gpu_id}] Procesadas: {processed_count}")
                            
            except Exception as e:
                print(f"⚠️ [GPU {gpu_id}] Error en {img_p.name}: {e}")

    except Exception as e:
        print(f"🔥 [GPU {gpu_id}] Error fatal: {e}")

def main():
    # USAR 'spawn' PARA EVITAR HERENCIA DE CONTEXTOS CUDA
    mp.set_start_method('spawn', force=True)

    sam3_ckpt = "/data/estudiantes/vacas/sam3/checkpoints/sam3.pt"
    input_dir = Path("runs/conteo_moscas")
    output_label_dir = Path("dataset_moscas/labels")
    output_label_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(input_dir.glob("roi_*.jpg")))
    
    manager = mp.Manager()
    task_queue = manager.Queue()
    for img_path in images:
        task_queue.put(img_path)
    for _ in AVAILABLE_GPUS:
        task_queue.put(None)

    print(f"🔥 Orquestación dinámica: {len(AVAILABLE_GPUS)} GPUs | {len(images)} imágenes.")

    processes = []
    for gpu_id in AVAILABLE_GPUS:
        p = mp.Process(target=worker_process, args=(gpu_id, task_queue, output_label_dir, sam3_ckpt))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n✅ DATASET COMPLETADO.")

if __name__ == "__main__":
    main()
