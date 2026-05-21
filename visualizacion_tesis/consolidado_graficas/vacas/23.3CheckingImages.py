# Script 23.3: Verificación Aleatoria de Etiquetas (Muestreo de 100)
# ==============================================================================
# COMANDO DE EJECUCIÓN (Copiar y pegar):
# nohup /opt/anaconda3/envs/tesis_vacas/bin/python 23.3CheckingImages.py > logs/output_23.3_checking.log 2>&1 &
# ==============================================================================

import os
import cv2
import random
import numpy as np
from pathlib import Path

def draw_yolo_polygons(image, label_path):
    """Lee un archivo YOLO de polígonos y los dibuja sobre la imagen."""
    h, w = image.shape[:2]
    if not label_path.exists():
        return image

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3: continue
        
        # class_id = parts[0]
        # El resto son pares x y normalizados
        coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
        # Des-normalizar
        coords[:, 0] *= w
        coords[:, 1] *= h
        pts = coords.astype(np.int32).reshape((-1, 1, 2))
        
        # Dibujar polígono en verde brillante
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    return image

def main():
    # --- CONFIGURACIÓN DE RUTAS ---
    img_dir = Path("runs/conteo_moscas")
    label_dir = Path("dataset_moscas/labels")
    existing_samples_dir = Path("verificacion_etiquetas/produccion_sam3")
    output_dir = Path("verificacion_etiquetas/muestreo_aleatorio_100")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Obtener nombres de archivos ya muestreados para excluirlos
    # El script anterior guardaba como 'sample_GPU_NAME.jpg' o 'qc_gpuX_fc_NAME.jpg'
    existing_sample_names = set()
    if existing_samples_dir.exists():
        for f in existing_samples_dir.glob("*.jpg"):
            # Intentamos encontrar el nombre original dentro del nombre de la muestra
            # Si el original es 'roi_ABC.jpg', la muestra es 'sample_2_roi_ABC.jpg'
            name = f.name
            if "roi_" in name:
                original_part = "roi_" + name.split("roi_")[-1]
                existing_sample_names.add(original_part)

    # 2. Buscar candidatos (Imágenes con etiquetas que NO han sido muestreadas)
    all_images = list(img_dir.glob("roi_*.jpg"))
    candidates = []

    print(f"🔍 Buscando candidatos entre {len(all_images)} imágenes...")

    for img_p in all_images:
        label_p = label_dir / f"{img_p.stem}.txt"
        
        # Debe tener etiqueta y no estar en el muestreo previo
        if label_p.exists() and img_p.name not in existing_sample_names:
            candidates.append(img_p)

    print(f"✅ Candidatos encontrados: {len(candidates)}")

    # 3. Seleccionar 100 al azar
    num_to_sample = min(100, len(candidates))
    selected_images = random.sample(candidates, num_to_sample)

    print(f"📸 Procesando {num_to_sample} imágenes aleatorias...")

    # 4. Generar visualizaciones
    for img_p in selected_images:
        label_p = label_dir / f"{img_p.stem}.txt"
        img_cv = cv2.imread(str(img_p))
        
        if img_cv is None: continue
        
        # Dibujar etiquetas
        img_viz = draw_yolo_polygons(img_cv, label_p)
        
        # Guardar en la nueva carpeta
        cv2.imwrite(str(output_dir / img_p.name), img_viz)

    print(f"🚀 ¡Listo! Revisa la carpeta: {output_dir}")

if __name__ == "__main__":
    main()
