import os
import random
from pathlib import Path

# Configuración
src_dir = Path("/data/estudiantes/vacas/dataset_moscas_v2/random_images/finalDatasetMoscas")
base_dest = Path("/data/estudiantes/vacas/dataset_moscas_v2/final_dataset_moscas")

# Proporciones
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

def create_split():
    # Listar imágenes
    images = sorted(list(src_dir.glob("*.jpg")))
    random.seed(42)  # Para reproducibilidad
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    print(f"📊 Total imágenes: {total}")
    for split_name, split_images in splits.items():
        dest_path = base_dest / split_name / "images"
        dest_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🔗 Creando links para {split_name} ({len(split_images)} imágenes)...")
        for img in split_images:
            link_name = dest_path / img.name
            if link_name.exists():
                link_name.unlink()
            # Crear link simbólico (relativo o absoluto)
            os.symlink(img, link_name)

    print("✅ Split completado con links simbólicos.")

if __name__ == "__main__":
    create_split()
