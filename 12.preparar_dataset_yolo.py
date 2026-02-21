import os
import random
from pathlib import Path
from tqdm import tqdm
import yaml

# NOTA DE PRODUCCIÓN: Este script procesa 174k archivos. 
# DEBE ejecutarse en segundo plano con:
# nohup /opt/anaconda3/envs/tesis_vacas/bin/python 12.preparar_dataset_yolo.py > logs/preparacion_dataset.log 2>&1 &

def create_dataset_structure(base_path):
    """Crea la jerarquía de carpetas para YOLOv11"""
    for split in ['train', 'val']:
        (base_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (base_path / split / 'labels').mkdir(parents=True, exist_ok=True)

def merge_labels(img_name, src_labels_fc, src_labels_kp, dest_label_path):
    """Une las etiquetas de Full Cow (Clase 0) y Key Parts (1, 2, 3)"""
    label_file = Path(img_name).with_suffix('.txt').name
    content = []
    
    # Leer Full Cow Seg (Clase 0)
    fc_path = src_labels_fc / label_file
    if fc_path.exists():
        with open(fc_path, 'r') as f:
            content.extend(f.readlines())
            
    # Leer Key Parts Seg (Clases 1, 2, 3)
    kp_path = src_labels_kp / label_file
    if kp_path.exists():
        with open(kp_path, 'r') as f:
            content.extend(f.readlines())
            
    if content:
        with open(dest_label_path, 'w') as f:
            f.writelines(content)
        return True
    return False

def main():
    # --- CONFIGURACIÓN ---
    SRC_IMG_DIR = Path("/data/estudiantes/vacas/dataset_curado/train/images")
    SRC_LABELS_FC = Path("/data/estudiantes/vacas/dataset_curado/train/labels/full_cow/seg")
    SRC_LABELS_KP = Path("/data/estudiantes/vacas/dataset_curado/train/labels/key_parts/seg")
    
    DEST_ROOT = Path("/data/estudiantes/vacas/dataset_final_yolo")
    VAL_SPLIT = 0.1 # 10% para validación
    
    print(f"🚀 Iniciando preparación del dataset en: {DEST_ROOT}")
    create_dataset_structure(DEST_ROOT)
    
    # Obtener lista de imágenes
    all_images = sorted(list(SRC_IMG_DIR.glob("*.jpg")))
    random.seed(42) # Para que el split sea reproducible
    random.shuffle(all_images)
    
    num_val = int(len(all_images) * VAL_SPLIT)
    val_images = all_images[:num_val]
    train_images = all_images[num_val:]
    
    splits = {'train': train_images, 'val': val_images}
    
    for split_name, images in splits.items():
        print(f"\n📦 Procesando split: {split_name} ({len(images)} imágenes)...")
        
        for img_path in tqdm(images):
            dest_img_path = DEST_ROOT / split_name / 'images' / img_path.name
            dest_lbl_path = DEST_ROOT / split_name / 'labels' / f"{img_path.stem}.txt"
            
            # 1. Crear Link Simbólico para la imagen (Seguridad de datos)
            if not dest_img_path.exists():
                try:
                    os.symlink(img_path, dest_img_path)
                except FileExistsError:
                    pass
            
            # 2. Fusionar etiquetas (Vaca + Partes)
            merge_labels(img_path.name, SRC_LABELS_FC, SRC_LABELS_KP, dest_lbl_path)

    # 3. Generar archivo dataset.yaml
    yaml_content = {
        'path': str(DEST_ROOT),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'cow',
            1: 'head',
            2: 'leg',
            3: 'tail'
        }
    }
    
    with open(DEST_ROOT / 'dataset.yaml', 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
        
    print(f"\n✅ Dataset preparado exitosamente.")
    print(f"📍 Ruta: {DEST_ROOT}")
    print(f"📄 Configuración: {DEST_ROOT / 'dataset.yaml'}")

if __name__ == "__main__":
    main()
