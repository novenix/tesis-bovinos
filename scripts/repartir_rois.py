import os
import shutil
from pathlib import Path

def distribute_rois():
    base_split_path = Path("/data/estudiantes/vacas/dataset_moscas_v2/final_dataset_moscas")
    roi_source_dir = Path("/data/estudiantes/vacas/runs/conteo_moscas_v25")
    
    splits = ['train', 'val', 'test']
    
    # 1. Mapear cada imagen original a su split
    image_to_split = {}
    for split in splits:
        img_dir = base_split_path / split / "images"
        if not img_dir.exists():
            continue
        for img_p in img_dir.glob("*.jpg"):
            image_to_split[img_p.stem] = split
            
    print(f"🔍 Mapeadas {len(image_to_split)} imágenes originales a sus splits.")

    # 2. Crear carpetas de destino para los ROIs
    for split in splits:
        (base_split_path / split / "rois").mkdir(parents=True, exist_ok=True)

    # 3. Clasificar y mover los ROIs
    all_rois = list(roi_source_dir.glob("roi_v25_*.jpg"))
    moved_count = 0
    
    print(f"📦 Procesando {len(all_rois)} ROIs...")

    for roi_p in all_rois:
        # El nombre es: roi_v25_{stem}_vaca{n}.jpg
        # Extraemos el stem eliminando el prefijo y el sufijo _vaca...
        name = roi_p.stem
        if not name.startswith("roi_v25_"):
            continue
            
        core_name = name.replace("roi_v25_", "")
        # Quitamos el "_vacaX" final. Buscamos la última aparición de "_vaca"
        if "_vaca" in core_name:
            original_stem = core_name.rsplit("_vaca", 1)[0]
            
            if original_stem in image_to_split:
                target_split = image_to_split[original_stem]
                dest_path = base_split_path / target_split / "rois" / roi_p.name
                
                # Copiamos en lugar de mover por seguridad, para que no pierdas nada si falla
                shutil.copy2(roi_p, dest_path)
                moved_count += 1

    print(f"✅ Se han repartido {moved_count} ROIs en las carpetas de train/val/test.")

if __name__ == "__main__":
    distribute_rois()
