import os
import cv2
import numpy as np
import random
from pathlib import Path

def draw_polygons():
    roi_dir = Path("runs/conteo_moscas_v25")
    label_dir = Path("dataset_moscas/labels_v25")
    viz_dir = Path("verificacion_etiquetas/produccion_v25_sam3")
    output_dir = Path("verificacion_etiquetas/muestras_extra_sam3")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Obtener archivos procesados
    all_rois = list(roi_dir.glob("*.jpg"))
    existing_viz = {f.name.replace("check_SUPER_", "") for f in viz_dir.glob("*.jpg")}
    
    # Filtrar para no repetir las que ya tienes
    candidates = [f for f in all_rois if f.name not in existing_viz]
    
    # Seleccionar 50 al azar que tengan etiquetas con contenido
    valid_candidates = []
    for f in candidates:
        label_f = label_dir / f"{f.stem}.txt"
        if label_f.exists() and os.path.getsize(label_f) > 0:
            valid_candidates.append(f)
    
    sample = random.sample(valid_candidates, min(50, len(valid_candidates)))
    
    print(f"🎨 Generando {len(sample)} nuevas visualizaciones en {output_dir}...")

    for img_p in sample:
        img = cv2.imread(str(img_p))
        h, w = img.shape[:2]
        label_f = label_dir / f"{img_p.stem}.txt"
        
        with open(label_f, "r") as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3: continue
            
            # Convertir coordenadas normalizadas a píxeles
            points = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            points[:, 0] *= w
            points[:, 1] *= h
            
            # Dibujar polígono en verde
            cv2.polylines(img, [points.astype(np.int32)], True, (0, 255, 0), 2)
            
        cv2.imwrite(str(output_dir / f"extra_viz_{img_p.name}"), img)

if __name__ == "__main__":
    draw_polygons()
