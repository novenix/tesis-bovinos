#!/usr/bin/env python3
"""
Pipeline de Curaduría Automatizada - VERSIÓN FINAL BLINDADA (1TB+)
================================================================================
Tesis de Maestría - Multi-GPU + Split Determinístico + Deduplicación + Vaciado
================================================================================
Cambios recientes:
- Implementación de Exclusión de Humanos (Filtro Negative Prompt 'person')
"""

import os
import sys
import subprocess
import shutil
import json
import torch
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
from tqdm import tqdm
from ultralytics import YOLO
import pandas as pd

# Autorizar carga de modelos de Ultralytics en PyTorch 2.6+
try:
    import torch.nn as nn
    from ultralytics.nn.tasks import WorldModel
    torch.serialization.add_safe_globals([WorldModel, nn.modules.container.Sequential, nn.modules.conv.Conv2d, nn.modules.batchnorm.BatchNorm2d, nn.modules.activation.SiLU, nn.modules.container.ModuleList])
except:
    pass

# ======================== CONFIGURACIÓN GLOBAL ================================
class Config:
    INPUT_DIR = Path("/data/estudiantes/vacas/datosCrudos")
    OUTPUT_DIR = Path("/data/estudiantes/vacas/dataset_curado")
    TEMP_DIR = Path("/data/estudiantes/vacas/frames_temp")
    LOG_DIR = Path("/data/estudiantes/vacas/logs")
    
    REGISTRY_FILE = OUTPUT_DIR / "registro_procesamiento.csv"

    FRAME_INTERVAL_SECONDS = 0.5 
    VIDEO_EXTENSIONS = ['.dav', '.avi', '.mp4', '.mkv']

    MODEL_NAME = "yolov8l-worldv2.pt"
    CONFIDENCE_THRESHOLD = 0.4
    
    # Clases de interés + "person" como filtro negativo
    TARGET_CLASSES = ["cow", "cattle", "bull", "calf", "livestock", "person"]
    CATTLE_CLASSES = ["cow", "cattle", "bull", "calf", "livestock"]

    TRAIN_RATIO = 0.8 # 80% Train, 20% Test (Permanente por fecha)
    VRAM_MIN_FREE = 4000  
    USE_GPU = True

# ======================== LÓGICA DE INTELIGENCIA ==============================

def get_split_by_date(date_str: str) -> str:
    hash_digest = hashlib.md5(date_str.encode()).hexdigest()
    hash_int = int(hash_digest[:8], 16) % 100
    return 'train' if hash_int < (Config.TRAIN_RATIO * 100) else 'test'

def get_video_date(path: Path) -> str:
    for part in path.parts:
        if len(part) == 10 and part.count('-') == 2:
            return part
    return "2000-01-01"

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

def process_video(video_path: Path, gpu_id: int) -> Dict:
    video_path = Path(video_path)
    video_stem = video_path.stem.replace(' ', '_')
    date_str = get_video_date(video_path)
    split = get_split_by_date(date_str)
    
    result = {
        'video_name': video_stem,
        'date': date_str,
        'split': split,
        'status': 'FAILED',
        'action_taken': 'NONE',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        if split == 'test':
            rel_path = video_path.relative_to(Config.INPUT_DIR)
            dest_path = Config.OUTPUT_DIR / 'test' / 'videos' / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            if dest_path.exists(): 
                os.remove(str(video_path))
                result['action_taken'] = 'BORRADO_DUPLICADO_TEST'
            else:
                shutil.move(str(video_path), str(dest_path))
                result['action_taken'] = 'MOVIDO_A_TEST'
            
            result['status'] = 'TERMINADO'
            return result

        # TRAIN: IA
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        video_temp_dir = Config.TEMP_DIR / f"tmp_{video_stem}_{gpu_id}_{os.getpid()}"
        video_temp_dir.mkdir(parents=True, exist_ok=True)

        cmd = ['ffmpeg', '-i', str(video_path), '-vf', f'fps=1/{Config.FRAME_INTERVAL_SECONDS}', '-qscale:v', '2', '-y', str(video_temp_dir / "f_%04d.jpg")]
        subprocess.run(cmd, capture_output=True, timeout=600)
        
        frame_paths = sorted(video_temp_dir.glob("*.jpg"))
        if frame_paths:
            # 2. Inferencia
            model = YOLO(Config.MODEL_NAME)
            model.set_classes(Config.TARGET_CLASSES)
            model.to("cuda:0")

            for frame_path in frame_paths:
                results = model.predict(source=str(frame_path), conf=Config.CONFIDENCE_THRESHOLD, verbose=False)
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    # LÓGICA DE EXCLUSIÓN DE HUMANOS
                    keep_frame = False
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_name = Config.TARGET_CLASSES[cls_id]
                        
                        # Si detectamos una de las clases de ganado, marcamos para guardar
                        if cls_name in Config.CATTLE_CLASSES:
                            keep_frame = True
                        
                        # Si detectamos una persona, evaluamos si descartar
                        # Regla: Si hay una persona presente con alta confianza, 
                        # solemos descartar para evitar falsos positivos de ganado
                        if cls_name == "person" and float(box.conf[0]) > 0.5:
                            keep_frame = False
                            break # Prioridad absoluta al humano (limpieza)

                    if keep_frame:
                        output_name = f"{date_str}_{video_stem}_{frame_path.stem}.jpg"
                        shutil.copy2(frame_path, Config.OUTPUT_DIR / 'train' / 'images' / output_name)
                
                frame_path.unlink()

            shutil.rmtree(video_temp_dir)
            os.remove(str(video_path))
            result['status'] = 'TERMINADO'
            result['action_taken'] = 'CURADO_Y_BORRADO'
        else:
            if video_temp_dir.exists(): shutil.rmtree(video_temp_dir)
            result['status'] = 'ERROR_EXTRACCION'
            
        return result

    except Exception as e:
        return {**result, 'status': 'ERROR', 'error': str(e)}

# ======================== MAIN ================================================

def main():
    for d in [Config.OUTPUT_DIR / 'train' / 'images', Config.OUTPUT_DIR / 'test' / 'videos', Config.TEMP_DIR, Config.LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    all_videos = sorted(list(Config.INPUT_DIR.rglob("*.dav")))
    if not all_videos:
        print("✅ No hay videos pendientes.")
        return

    registry = load_registry()
    to_process = []
    
    for v in all_videos:
        v_name = v.stem.replace(' ', '_')
        v_date = get_video_date(v)
        if not is_already_processed(v_name, v_date, registry):
            to_process.append(v)
        else:
            try: os.remove(str(v)) # Limpieza de duplicados ya registrados
            except: pass

    if not to_process:
        print("✅ Carpeta datosCrudos vacía.")
        return

    print(f"🚀 Iniciando Pipeline con Exclusión de Humanos ({len(to_process)} videos)...")

    try:
        cmd_gpu = "nvidia-smi --query-gpu=index,memory.free --format=csv,nounits,noheader"
        output = subprocess.check_output(cmd_gpu, shell=True).decode('utf-8')
        available_gpus = [int(line.split(',')[0]) for line in output.strip().split('\n') if int(line.split(',')[1]) >= Config.VRAM_MIN_FREE]
    except: available_gpus = []
    
    num_workers = len(available_gpus) if available_gpus else 2
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_video, v, available_gpus[i % len(available_gpus)] if available_gpus else 0): v for i, v in enumerate(to_process)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Vaciando datosCrudos"):
            res = future.result()
            df_current = load_registry()
            df_current = pd.concat([df_current, pd.DataFrame([res])], ignore_index=True)
            df_current.to_csv(Config.REGISTRY_FILE, index=False)

    subprocess.run(f"find {Config.INPUT_DIR} -type d -empty -delete", shell=True)
    print(f"✅ ¡Proceso completado! Dataset libre de humanos (aprox).")

if __name__ == "__main__":
    main()