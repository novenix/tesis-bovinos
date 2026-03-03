import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from multiprocessing import Pool

def simplify_polygon(coords, tolerance=0.001):
    poly = np.array(coords).reshape((-1, 1, 2)).astype(np.float32)
    epsilon = tolerance * cv2.arcLength(poly, True)
    simplified = cv2.approxPolyDP(poly, epsilon, True)
    return simplified.reshape(-1, 2).flatten().tolist()

def process_single_file(args):
    file_path, output_path = args
    new_lines = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                class_id = parts[0]
                coords = [float(x) for x in parts[1:]]
                pairs = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                simplified_coords = simplify_polygon(pairs)
                new_line = f"{class_id} " + " ".join([f"{x:.6f}" for x in simplified_coords])
                new_lines.append(new_line)
        
        with open(output_path / file_path.name, "w") as f:
            f.write("\n".join(new_lines))
        return True
    except:
        return False

def setup_v2_dataset(base_original, base_v2, split, num_cores):
    orig_images = Path(base_original) / split / "images"
    orig_labels = Path(base_original) / split / "labels"
    v2_images = Path(base_v2) / split / "images"
    v2_labels = Path(base_v2) / split / "labels"
    
    v2_images.mkdir(parents=True, exist_ok=True)
    v2_labels.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Preparando {split}...")
    
    image_files = list(orig_images.glob("*.jpg"))
    for img in tqdm(image_files, desc=f"Enlazando imágenes {split}"):
        target = v2_images / img.name
        if not target.exists():
            os.symlink(img, target)
            
    label_files = list(orig_labels.glob("*.txt"))
    tasks = [(f, v2_labels) for f in label_files]
    
    print(f"🚀 Simplificando etiquetas de {split} usando {num_cores} núcleos...")
    with Pool(num_cores) as p:
        list(tqdm(p.imap_unordered(process_single_file, tasks), total=len(tasks), desc=f"Procesando {split}"))

if __name__ == "__main__":
    ORIGINAL_PATH = "/data/estudiantes/vacas/dataset_final_yolo"
    V2_PATH = "/data/estudiantes/vacas/dataset_v2_simplified"
    NUM_CORES = 37
    
    setup_v2_dataset(ORIGINAL_PATH, V2_PATH, "train", NUM_CORES)
    setup_v2_dataset(ORIGINAL_PATH, V2_PATH, "val", NUM_CORES)
    
    yaml_content = f"path: {V2_PATH}\ntrain: train/images\nval: val/images\nnames:\n  0: cow\n  1: head\n  2: leg\n  3: tail\n"
    with open(f"{V2_PATH}/dataset_v2.yaml", "w") as f:
        f.write(yaml_content)
        
    print(f"\n✅ Dataset V2 terminado con éxito (37 núcleos utilizados).")
    print(f"📍 YAML para entrenamiento: {V2_PATH}/dataset_v2.yaml")
