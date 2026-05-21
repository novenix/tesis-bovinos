#!/usr/bin/env python3
"""
Script 2: Pipeline de Curaduría (Versión 2-Pasos)
================================================
Este script asume que los datos ya están organizados en datosCrudos/train y datosCrudos/test.
- TRAIN: Inferencia IA + Filtro Humanos -> dataset_curado/train/images/
- TEST: Creación de Enlaces Simbólicos -> dataset_curado/test/videos/
- NO BORRA NADA DE LOS DATOS ORIGINALES.
"""

import os
import sys
import subprocess
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm

# ======================== CONFIGURACIÓN =======================================
class Config:
    # Entradas (Organizadas por Script 1)
    INPUT_TRAIN = Path("/data/estudiantes/vacas/datosCrudos/train")
    INPUT_TEST = Path("/data/estudiantes/vacas/datosCrudos/test")
    
    # Salidas
    OUTPUT_DIR = Path("/data/estudiantes/vacas/dataset_curado")
    TEMP_DIR = Path("/data/estudiantes/vacas/frames_temp")
    
    REGISTRY_FILE = OUTPUT_DIR / "registro_procesamiento.csv"

    FRAME_INTERVAL_SECONDS = 0.5 
    VIDEO_EXTENSIONS = ['.dav', '.avi', '.mp4', '.mkv']

    MODEL_NAME = "yolov8l-worldv2.pt"
    CONFIDENCE_THRESHOLD = 0.4
    TARGET_CLASSES = ["cow", "cattle", "bull", "calf", "livestock", "person"]
    CATTLE_CLASSES = ["cow", "cattle", "bull", "calf", "livestock"]

    USE_GPU = True
    VRAM_MIN_FREE = 4000

# ======================== UTILIDADES ==========================================

def get_video_date(path: Path) -> str:
    """Extrae fecha de la ruta o nombre."""
    import re
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", str(path))
    return date_match.group(1) if date_match else "2000-01-01"

def load_registry() -> pd.DataFrame:
    if Config.REGISTRY_FILE.exists():
        try: return pd.read_csv(Config.REGISTRY_FILE)
        except: pass
    return pd.DataFrame(columns=['video_name', 'date', 'split', 'status', 'action_taken', 'timestamp'])

def is_already_processed(video_name: str, date_str: str, registry: pd.DataFrame) -> bool:
    if registry.empty: return False
    match = registry[(registry['video_name'] == video_name) & (registry['date'] == date_str) & (registry['status'] == 'TERMINADO')]
    return not match.empty

# ======================== PROCESAMIENTO =======================================

def process_test_video(video_path: Path) -> Dict:
    """Crea un link simbólico para videos de test."""
    video_stem = video_path.stem.replace(' ', '_')
    date_str = get_video_date(video_path)
    
    # Ruta relativa dentro de test/ para mantener subcarpetas
    rel_path = video_path.relative_to(Config.INPUT_TEST)
    dest_path = Config.OUTPUT_DIR / 'test' / 'videos' / rel_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        'video_name': video_stem, 'date': date_str, 'split': 'test',
        'status': 'TERMINADO', 'action_taken': 'SYMLINK_CREATED',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        if dest_path.is_symlink() or dest_path.exists():
            result['action_taken'] = 'SYMLINK_ALREADY_EXISTS'
        else:
            # Crear el enlace simbólico (referencia)
            os.symlink(video_path, dest_path)
    except Exception as e:
        result['status'] = 'ERROR'
        result['error'] = str(e)
    
    return result

def process_train_video(video_path: Path, gpu_id: int) -> Dict:
    """Procesa video de train con IA."""
    video_stem = video_path.stem.replace(' ', '_')
    date_str = get_video_date(video_path)
    
    result = {
        'video_name': video_stem, 'date': date_str, 'split': 'train',
        'status': 'FAILED', 'action_taken': 'NONE',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        video_temp_dir = Config.TEMP_DIR / f"tmp_{video_stem}_{os.getpid()}"
        video_temp_dir.mkdir(parents=True, exist_ok=True)

        # 1. Extracción de frames
        cmd = ['ffmpeg', '-i', str(video_path), '-vf', f'fps=1/{Config.FRAME_INTERVAL_SECONDS}', '-qscale:v', '2', '-y', str(video_temp_dir / "f_%04d.jpg")]
        subprocess.run(cmd, capture_output=True, timeout=600)
        
        frame_paths = sorted(video_temp_dir.glob("*.jpg"))
        if not frame_paths:
            result['status'] = 'ERROR_EXTRACCION'
            shutil.rmtree(video_temp_dir)
            return result

        # 2. IA
        model = YOLO(Config.MODEL_NAME)
        model.set_classes(Config.TARGET_CLASSES)
        
        frames_saved = 0
        for frame_path in frame_paths:
            results = model.predict(source=str(frame_path), conf=Config.CONFIDENCE_THRESHOLD, verbose=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                keep_frame = False
                for box in results[0].boxes:
                    cls_name = Config.TARGET_CLASSES[int(box.cls[0])]
                    if cls_name in Config.CATTLE_CLASSES: keep_frame = True
                    if cls_name == "person" and float(box.conf[0]) > 0.5:
                        keep_frame = False
                        break 

                if keep_frame:
                    output_name = f"{date_str}_{video_stem}_{frame_path.stem}.jpg"
                    dest_img = Config.OUTPUT_DIR / 'train' / 'images' / output_name
                    shutil.copy2(frame_path, dest_img)
                    frames_saved += 1
                elif any(Config.TARGET_CLASSES[int(box.cls[0])] == "person" and float(box.conf[0]) > 0.5 for box in results[0].boxes):
                    # Opcional: imprimir en consola si hay humanos para debuggear
                    # print(f"  ⚠️ Frame {frame_path.name} descartado: Presencia de Humano detectada.")
                    pass
            
            frame_path.unlink()

        shutil.rmtree(video_temp_dir)
        result['status'] = 'TERMINADO'
        result['action_taken'] = f'CURADO_EXITOSO_{frames_saved}_IMG'
        return result

    except Exception as e:
        result['status'] = 'ERROR'
        result['error'] = str(e)
        return result

# ======================== MAIN ================================================

def main():
    # Asegurar carpetas
    for d in [Config.OUTPUT_DIR / 'train' / 'images', Config.OUTPUT_DIR / 'test' / 'videos', Config.TEMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    registry = load_registry()
    
    # 1. Identificar videos de TEST (Rápido, solo links)
    test_videos = []
    if Config.INPUT_TEST.exists():
        for ext in Config.VIDEO_EXTENSIONS:
            test_videos.extend(list(Config.INPUT_TEST.rglob(f"*{ext}")))
    
    print(f"🔗 Creando referencias para {len(test_videos)} videos de TEST...")
    for v in tqdm(test_videos, desc="Links de Test"):
        v_name = v.stem.replace(' ', '_')
        v_date = get_video_date(v)
        if not is_already_processed(v_name, v_date, registry):
            res = process_test_video(v)
            registry = pd.concat([registry, pd.DataFrame([res])], ignore_index=True)
            registry.to_csv(Config.REGISTRY_FILE, index=False)

    # 2. Identificar videos de TRAIN (Lento, requiere GPU)
    train_videos = []
    if Config.INPUT_TRAIN.exists():
        for ext in Config.VIDEO_EXTENSIONS:
            train_videos.extend(list(Config.INPUT_TRAIN.rglob(f"*{ext}")))
    
    to_process_train = [v for v in train_videos if not is_already_processed(v.stem.replace(' ', '_'), get_video_date(v), registry)]

    if not to_process_train:
        print("✅ No hay videos de TRAIN pendientes.")
        return

    print(f"🚀 Procesando {len(to_process_train)} videos de TRAIN con IA...")

    # Gestionar GPUs
    try:
        cmd_gpu = "nvidia-smi --query-gpu=index,memory.free --format=csv,nounits,noheader"
        output = subprocess.check_output(cmd_gpu, shell=True).decode('utf-8')
        available_gpus = [int(line.split(',')[0]) for line in output.strip().split('\n') if int(line.split(',')[1]) >= Config.VRAM_MIN_FREE]
    except: available_gpus = [0]
    
    num_workers = len(available_gpus) if available_gpus else 2
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_train_video, v, available_gpus[i % len(available_gpus)]): v for i, v in enumerate(to_process_train)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Curaduría IA"):
            res = future.result()
            # Recargar registro para evitar conflictos entre procesos
            curr_registry = load_registry()
            curr_registry = pd.concat([curr_registry, pd.DataFrame([res])], ignore_index=True)
            curr_registry.to_csv(Config.REGISTRY_FILE, index=False)

    print(f"\n✨ ¡Pipeline completado! Resultados en {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
