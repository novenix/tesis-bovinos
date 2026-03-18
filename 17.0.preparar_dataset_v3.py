
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def simplify_polygon(coords, target_points=250):
    """
    Simplifica un polígono usando Douglas-Peucker buscando acercarse a target_points.
    """
    pts = np.array(coords).reshape(-1, 2)
    
    # Búsqueda binaria simple para el epsilon óptimo
    low = 0.0
    high = 0.01
    best_epsilon = 0.001
    
    for _ in range(10):  # 10 iteraciones para ajustar epsilon
        mid = (low + high) / 2
        epsilon = mid * cv2.arcLength(pts.astype(np.float32), True)
        approx = cv2.approxPolyDP(pts.astype(np.float32), epsilon, True)
        
        if len(approx) > target_points:
            low = mid
        else:
            high = mid
            best_epsilon = mid
            
    # Resultado final con el mejor epsilon encontrado
    final_epsilon = best_epsilon * cv2.arcLength(pts.astype(np.float32), True)
    approx = cv2.approxPolyDP(pts.astype(np.float32), final_epsilon, True)
    return approx.reshape(-1).tolist()

def process_labels(src_dir, dst_dir, target_pts=250):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    label_files = list(src_dir.glob("*.txt"))
    print(f"📄 Procesando {len(label_files)} archivos de etiquetas en {src_dir.name}...")
    
    for label_file in tqdm(label_files):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7: continue  # No es un polígono válido
            
            cls = parts[0]
            coords = [float(x) for x in parts[1:]]
            
            # Solo simplificar si tiene más puntos que el objetivo
            if len(coords) / 2 > target_pts:
                coords = simplify_polygon(coords, target_pts)
            
            new_line = f"{cls} " + " ".join([f"{x:.6f}" for x in coords]) + "\n"
            new_lines.append(new_line)
            
        with open(dst_dir / label_file.name, 'w') as f:
            f.writelines(new_lines)

def main():
    base_path = Path("/data/estudiantes/vacas")
    src_dataset = base_path / "dataset_final_yolo"
    dst_dataset = base_path / "dataset_v3_medium"
    
    # 1. Crear estructura de carpetas
    for split in ["train", "val"]:
        (dst_dataset / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_dataset / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # 2. Enlaces simbólicos para imágenes (Ahorro de espacio)
        print(f"🔗 Creando symlinks para imágenes de {split}...")
        src_images = src_dataset / split / "images"
        dst_images = dst_dataset / split / "images"
        
        for img in src_images.glob("*.jpg"):
            target = dst_images / img.name
            if not target.exists():
                os.symlink(img, target)
        
        # 3. Procesar y simplificar etiquetas
        process_labels(src_dataset / split / "labels", dst_dataset / split / "labels", target_pts=250)

    # 4. Crear archivo YAML para el modelo Medium
    yaml_content = f"""path: {dst_dataset}
train: train/images
val: val/images

names:
  0: cow
  1: head
  2: leg
  3: tail
"""
    with open(dst_dataset / "dataset_v3.yaml", 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Dataset V3 (Medium) listo en: {dst_dataset}")
    print(f"📝 YAML creado en: {dst_dataset / 'dataset_v3.yaml'}")

if __name__ == "__main__":
    main()
