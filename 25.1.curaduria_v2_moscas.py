#!/usr/bin/env python3
"""
Script 25.1: Curaduría de Imágenes de Moscas (Basado en Detección de Vacas)
========================================================================
Este script filtra imágenes del dataset de moscas v2 basándose en:
1. Presencia de Ganado (Vacas, etc.) -> Mantener
2. Imágenes vacías o con ruido -> Descartar

Modelo: YOLO-World v2 (Open Vocabulary)
PARALELISMO: Multi-GPU (GPUs 2, 3, 4, 5, 6, 7)

EJECUCIÓN (Segundo Plano y Paralelo):
------------------------------------
Para ejecutar en segundo plano y capturar la salida:
nohup python3 25.1.curaduria_v2_moscas.py > logs/curaduria_v2_run.log 2>&1 &

El script ya maneja internamente el paralelismo entre las GPUs configuradas.
"""

import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm

# ======================== CONFIGURACIÓN =======================================
class Config:
    # Rutas
    INPUT_DIR = Path("/data/estudiantes/vacas/dataset_moscas_v2/random_images/randomimages")
    OUTPUT_DIR = Path("/data/estudiantes/vacas/dataset_moscas_v2/random_images/finalDatasetMoscas")
    
    # Registro de resultados
    LOG_DIR = Path("/data/estudiantes/vacas/logs")
    REGISTRY_FILE = LOG_DIR / f"curaduria_moscas_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # Detección
    MODEL_NAME = "yolov8l-worldv2.pt"
    CONFIDENCE_THRESHOLD = 0.4
    
    TARGET_CLASSES = ["cow", "cattle", "bull", "calf", "livestock"]
    CATTLE_CLASSES = ["cow", "cattle", "bull", "calf", "livestock"]

    # Hardware (Configuración solicitada: GPUs 2 a 7)
    AVAILABLE_GPUS = [2, 3, 4, 5, 6, 7]
    BATCH_SIZE = 1 

# ======================== PROCESAMIENTO =======================================

def process_image_batch(image_paths, gpu_id):
    """Procesa un lote de imágenes en una GPU específica."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Evitar fragmentación de memoria
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    results_list = []
    try:
        # Forzar recarga del modelo en cada worker para asegurar que use la GPU asignada
        model = YOLO(Config.MODEL_NAME)
        model.set_classes(Config.TARGET_CLASSES)
        
        for img_path in image_paths:
            # Inferencia
            results = model.predict(source=str(img_path), conf=Config.CONFIDENCE_THRESHOLD, verbose=False)
            
            keep_image = False
            reason = "NO_CATTLE"
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                keep_image = True
                reason = "CATTLE_DETECTED"
            
            if keep_image:
                dest_path = Config.OUTPUT_DIR / img_path.name
                shutil.copy2(img_path, dest_path)
            
            results_list.append({
                'image_name': img_path.name,
                'status': 'KEEP' if keep_image else 'DISCARD',
                'reason': reason,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
    except Exception as e:
        for img_path in image_paths:
            results_list.append({
                'image_name': img_path.name,
                'status': 'ERROR',
                'reason': str(e),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
    return results_list

# ======================== MAIN ================================================

def main():
    print(f"🚀 Iniciando Curaduría V2 (Detección de Vacas)")
    print(f"📂 Origen: {Config.INPUT_DIR}")
    print(f"📂 Destino: {Config.OUTPUT_DIR}")

    # Asegurar carpetas
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Listar imágenes
    all_images = sorted(list(Config.INPUT_DIR.glob("*.jpg")))
    print(f"📸 Total imágenes a procesar: {len(all_images)}")

    if not all_images:
        print("❌ No se encontraron imágenes .jpg en el directorio de origen.")
        return

    # Usar las GPUs configuradas
    available_gpus = Config.AVAILABLE_GPUS
    num_workers = len(available_gpus)
    print(f"🤖 Usando {num_workers} GPUs en paralelo: {available_gpus}")

    # Dividir en lotes para paralelismo
    # Dividimos en más lotes que workers para mantener las GPUs ocupadas uniformemente
    batch_size = max(1, len(all_images) // (num_workers * 20)) 
    batches = [all_images[i:i + batch_size] for i in range(0, len(all_images), batch_size)]
    
    full_registry = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, batch in enumerate(batches):
            # Asignar rotativamente cada lote a una GPU
            gpu_id = available_gpus[i % len(available_gpus)]
            futures.append(executor.submit(process_image_batch, batch, gpu_id))

        with tqdm(total=len(all_images), desc="Curando Dataset") as pbar:
            for future in as_completed(futures):
                batch_results = future.result()
                full_registry.extend(batch_results)
                pbar.update(len(batch_results))
                
                # Guardado incremental del registro
                pd.DataFrame(full_registry).to_csv(Config.REGISTRY_FILE, index=False)

    # Resumen final
    df = pd.DataFrame(full_registry)
    keep_count = len(df[df['status'] == 'KEEP'])
    discard_count = len(df[df['status'] == 'DISCARD'])
    error_count = len(df[df['status'] == 'ERROR'])

    print(f"\n✨ Proceso Completado!")
    print(f"✅ Imágenes mantenidas: {keep_count}")
    print(f"❌ Imágenes descartadas: {discard_count}")
    if error_count > 0:
        print(f"⚠️ Errores encontrados: {error_count}")
    print(f"📊 Registro guardado en: {Config.REGISTRY_FILE}")

if __name__ == "__main__":
    main()
