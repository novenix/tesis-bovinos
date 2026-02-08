#!/usr/bin/env python3
"""
Filtro de Redundancia Temporal - Curaduría de Datos
===================================================
Analiza las imágenes en dataset_curado y elimina aquellas que son casi
idénticas (donde la vaca no se ha movido), basándose en SSIM.
"""

import os
import shutil
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from tqdm import tqdm

# --- CONFIGURACIÓN ---
DATASET_DIR = Path("/data/estudiantes/vacas/dataset_curado")
# Umbral de similitud (0.0 a 1.0)
# 0.95 significa que si son 95% iguales, se considera duplicado.
SIMILARITY_THRESHOLD = 0.95

def get_video_id(filename):
    """Extrae el nombre del video original del nombre del archivo"""
    # Formato: video_name_timestamp_frame_0001.jpg
    # Buscamos la parte antes del timestamp (8 digitos _ 6 digitos)
    parts = filename.split('_')
    if len(parts) >= 3:
        return "_".join(parts[:-3])
    return filename

def process_cleaning():
    print(f"🔍 Analizando redundancia en: {DATASET_DIR}")
    
    files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.jpg')])
    if not files:
        print("❌ No hay imágenes para analizar.")
        return

    # Agrupar por video
    video_groups = {}
    for f in files:
        vid = get_video_id(f)
        if vid not in video_groups: video_groups[vid] = []
        video_groups[vid].append(f)

    to_delete = []
    total_saved = 0

    print(f"📦 Videos detectados: {len(video_groups)}")
    
    for vid, frames in video_groups.items():
        if len(frames) < 2: continue
        
        # Siempre nos quedamos con el primer frame
        last_frame_path = DATASET_DIR / frames[0]
        last_img = cv2.imread(str(last_frame_path), cv2.IMREAD_GRAYSCALE)
        # Redimensionar para que la comparacion sea ultra rapida
        last_img = cv2.resize(last_img, (256, 256))

        for i in range(1, len(frames)):
            current_frame_path = DATASET_DIR / frames[i]
            curr_img = cv2.imread(str(current_frame_path), cv2.IMREAD_GRAYSCALE)
            curr_img_resized = cv2.resize(curr_img, (256, 256))

            # Calcular similitud estructural
            score, _ = ssim(last_img, curr_img_resized, full=True)

            if score > SIMILARITY_THRESHOLD:
                # Es un duplicado (vaca quieta)
                to_delete.append(current_frame_path)
            else:
                # La vaca se movió suficiente, este frame es el nuevo referente
                last_img = curr_img_resized
                total_saved += 1

    print("
" + "="*50)
    print(f"📊 RESULTADOS DEL ANÁLISIS:")
    print(f"   Total imágenes analizadas: {len(files)}")
    print(f"   Imágenes redundantes (vaca quieta): {len(to_delete)}")
    print(f"   Imágenes únicas (vaca moviéndose): {len(files) - len(to_delete)}")
    print(f"   Ahorro potencial de espacio: {(len(to_delete)/len(files))*100:.1f}%")
    print("="*50)

    if to_delete:
        confirm = input(f"
⚠️ ¿Deseas BORRAR permanentemente las {len(to_delete)} imágenes redundantes? (si/no): ")
        if confirm.lower() == 'si':
            for p in tqdm(to_delete, desc="Borrando"):
                try:
                    p.unlink()
                except: pass
            print(f"
✅ Limpieza completada.")
        else:
            print("
❌ Operación cancelada. No se borró nada.")

if __name__ == "__main__":
    process_cleaning()
