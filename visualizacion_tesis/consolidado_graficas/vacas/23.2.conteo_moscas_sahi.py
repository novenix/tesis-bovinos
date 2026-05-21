# Script 18: Pipeline de Inferencia de Dos Estadios (Detección de Bovino + Conteo de Moscas SAHI)
# Referencia: ContextoMoscas.pdf - Propuesta Arquitectónica Definitiva
# Autor: Gemini CLI (Basado en requerimientos del Proyecto de Grado)

"""
REQUISITOS PREVIOS EN CRATOS:
1. Instalar SAHI: 
   /opt/anaconda3/envs/tesis_vacas/bin/pip install sahi
2. Tener los pesos:
   - yolo26n-seg.pt (Stage 1: Detección de Bovino)
   - yolo26m-seg.pt (Stage 2: Detección de Moscas)
"""

# COMANDO EJECUTADO EN SESIÓN (MARZO 30, 2026):
# nohup bash -c "/opt/anaconda3/envs/tesis_vacas/bin/python -u /data/estudiantes/vacas/23.2.conteo_moscas_sahi.py" > logs/output_v23.2_sahi.log 2>&1 &

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import threading
import time
import psutil
import gc

# Intentar importar SAHI (Librería recomendada en la pág. 9 del PDF)
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.utils.cv import read_image
except ImportError:
    print("⚠️  ADVERTENCIA: SAHI no está instalado. Ejecute: pip install sahi")
    print("El script intentará usar inferencia estándar si SAHI falla.")

from ultralytics import YOLO

# --- CONFIGURACIÓN DE AMBIENTE CRATOS ---
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
# Seleccionar GPUs solicitadas: del 2 al 7
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7" 

# --- CLASE VIGILANTE DE RECURSOS (ESTÁNDAR DEL PROYECTO) ---
class CratosWatcher(threading.Thread):
    def __init__(self, log_path, interval=10):
        super().__init__()
        self.log_path = log_path
        self.interval = interval
        self.daemon = True
        self.stop_event = threading.Event()

    def run(self):
        with open(self.log_path, "a") as f:
            f.write(f"\n--- INICIO PIPELINE MOSCAS {datetime.now()} ---\n")
            process = psutil.Process(os.getpid())
            while not self.stop_event.is_set():
                try:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    mem = psutil.virtual_memory()
                    proc_mem = process.memory_info().rss / (1024**3)
                    f.write(f"{datetime.now().strftime('%H:%M:%S')} | RAM: {mem.used/(1024**3):.1f}GB | Proc: {proc_mem:.1f}GB\n")
                    f.flush()
                except: pass
                time.sleep(self.interval)

# --- PIPELINE DE DOS ESTADIOS ---
class FlyCounterPipeline:
    def __init__(self, stage1_weights, stage2_weights, device='cuda'):
        print("🕒 Cargando modelos en memoria...")
        # Estadio 1: Macro-Escala (Bovino) - Nano para velocidad
        self.model_stage1 = YOLO(stage1_weights)
        
        # Estadio 2: Micro-Escala (Moscas) - Medium para precisión
        # Usamos el wrapper de SAHI para el Estadio 2
        self.model_stage2 = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=stage2_weights,
            confidence_threshold=0.25,
            device=device,
        )
        self.device = device
        print("✅ Modelos listos.")

    def run_inference(self, image_path, output_dir):
        img = cv2.imread(str(image_path))
        if img is None: return
        h, w = img.shape[:2]
        
        # --- ESTADIO 1: DETECCIÓN DEL BOVINO ---
        # Buscamos la clase 'cow' (ID 19 en COCO, pero depende del entrenamiento previo)
        # Asumimos que el modelo V17 detecta 'cow' o 'lomo'.
        results_s1 = self.model_stage1.predict(img, conf=0.5, verbose=False)[0]
        
        if len(results_s1.boxes) == 0:
            print(f"❌ No se detectó bovino en {image_path.name}")
            return 0

        # Tomamos el bovino con mayor confianza
        best_box = results_s1.boxes[0]
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        
        # Extraer ROI del lomo (Crop de alta resolución)
        # El PDF sugiere que el ROI es vital para no perder resolución
        roi = img[y1:y2, x1:x2].copy()
        roi_path = output_dir / f"roi_{image_path.name}"
        cv2.imwrite(str(roi_path), roi)

        # --- ESTADIO 2: CONTEO DE MOSCAS (SAHI) ---
        # Aplicamos Tiling para no destruir la mosca sub-píxel
        # Parámetros basados en pág 18 del PDF: Slices de 512x512 o 1024x1024
        result_s2 = get_sliced_prediction(
            roi, # Pasar el array directamente
            self.model_stage2,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0 # Silenciar SAHI
        )
        
        fly_count = len(result_s2.object_prediction_list)
        
        # --- VISUALIZACIÓN DETALLADA ---
        # 1. Dibujar cada mosca en el recorte (ROI) para ver el detalle máximo
        roi_annotated = roi.copy()
        for prediction in result_s2.object_prediction_list:
            bbox = prediction.bbox.to_voc_bbox() # [xmin, ymin, xmax, ymax]
            # Dibujar un cuadrito o círculo pequeño sobre la mosca en el ROI
            cv2.rectangle(roi_annotated, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            
            # 2. Dibujar también sobre la imagen completa (ajustando coordenadas)
            # El ROI empieza en (x1, y1), así que sumamos eso
            fx1, fy1, fx2, fy2 = int(bbox[0]+x1), int(bbox[1]+y1), int(bbox[2]+x1), int(bbox[3]+y1)
            cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)

        # Dibujar conteo y vaca en la imagen original
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, f"Moscas: {fly_count}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        res_path = output_dir / f"res_{image_path.name}"
        roi_path = output_dir / f"roi_{image_path.name}"
        cv2.imwrite(str(res_path), img)
        cv2.imwrite(str(roi_path), roi_annotated)
        
        print(f"📌 {image_path.name} -> Conteo: {fly_count} moscas.")
        return fly_count

def main():
    # Rutas
    weights_s1 = "yolo26n-seg.pt"
    weights_s2 = "yolo26m-seg.pt"
    input_images = Path("dataset_final_yolo/val/images") # Usar imágenes de validación
    output_dir = Path("runs/conteo_moscas")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Telemetría
    log_path = Path("logs/telemetria_v23.2_conteo.log")
    Path("logs").mkdir(exist_ok=True)
    watcher = CratosWatcher(log_path)
    watcher.start()

    # Inicializar Pipeline
    try:
        pipeline = FlyCounterPipeline(weights_s1, weights_s2)
    except Exception as e:
        print(f"🔥 Error inicializando SAHI/YOLO: {e}")
        return

    # Procesar imágenes
    images = list(input_images.glob("*.jpg")) + list(input_images.glob("*.png"))
    if not images:
        print(f"⚠️ No hay imágenes en {input_images}")
        return

    total_flies = 0
    print(f"🚀 Iniciando procesamiento de {len(images)} imágenes...")
    
    start_time = time.time()
    for img_p in images:
        count = pipeline.run_inference(img_p, output_dir)
        total_flies += (count if count else 0)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / len(images)
    
    print("\n--- RESUMEN FINAL ---")
    print(f"Total imágenes: {len(images)}")
    print(f"Total moscas detectadas: {total_flies}")
    print(f"Tiempo promedio por imagen: {avg_time:.3f}s")
    print(f"Resultados guardados en: {output_dir}")

if __name__ == "__main__":
    main()
