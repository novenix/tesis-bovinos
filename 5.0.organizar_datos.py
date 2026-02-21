#!/usr/bin/env python3
"""
Script 5.0: Organización Equilibrada (70/30)
===========================================
Organiza los videos de la raíz en datosCrudos/train y datosCrudos/test.
- Hace el split ARCHIVO POR ARCHIVO para garantizar el 70/30 real.
- Recrea la estructura de carpetas original dentro de train/ y test/.
- Mueve los archivos (no los borra).
"""

import os
import shutil
import hashlib
from pathlib import Path
from tqdm import tqdm

class Config:
    BASE_DIR = Path("/data/estudiantes/vacas/datosCrudos")
    TRAIN_DIR = BASE_DIR / "train"
    TEST_DIR = BASE_DIR / "test"
    
    # Nuevo Ratio solicitado: 70% Train, 30% Test
    TRAIN_RATIO = 0.7
    VIDEO_EXTENSIONS = ['.dav', '.avi', '.mp4', '.mkv']

def get_split_by_filename(filename: str) -> str:
    """Decide el split basado en el nombre del archivo (determinístico)."""
    hash_digest = hashlib.md5(filename.encode()).hexdigest()
    hash_int = int(hash_digest[:8], 16) % 100
    return 'train' if hash_int < (Config.TRAIN_RATIO * 100) else 'test'

def main():
    Config.TRAIN_DIR.mkdir(exist_ok=True)
    Config.TEST_DIR.mkdir(exist_ok=True)

    # 1. Buscar todos los videos en la raíz (excluyendo carpetas train/test)
    all_videos = []
    for ext in Config.VIDEO_EXTENSIONS:
        # Buscamos de forma recursiva pero filtramos para no entrar en train/test
        for path in Config.BASE_DIR.rglob(f"*{ext}"):
            if "train" not in path.parts and "test" not in path.parts:
                all_videos.append(path)

    if not all_videos:
        print("✅ No hay videos nuevos para organizar en la raíz.")
        return

    print(f"📦 Se encontraron {len(all_videos)} videos. Repartiendo 70/30...")

    for video_path in tqdm(all_videos, desc="Repartiendo archivos"):
        # 2. Determinar destino
        split = get_split_by_filename(video_path.name)
        target_root = Config.TRAIN_DIR if split == 'train' else Config.TEST_DIR
        
        # 3. Calcular ruta relativa para recrear estructura
        # Si el video está en datosCrudos/Día1/Cam1/v.dav -> relativo es Día1/Cam1/v.dav
        rel_path = video_path.relative_to(Config.BASE_DIR)
        dest_path = target_root / rel_path
        
        # 4. Mover con seguridad
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not dest_path.exists():
                shutil.move(str(video_path), str(dest_path))
            else:
                print(f"⚠️ Ya existe: {rel_path} en {split}. Omitiendo.")
        except Exception as e:
            print(f"❌ Error moviendo {video_path.name}: {e}")

    # 5. Limpieza de carpetas vacías en la raíz
    print("\n🧹 Limpiando carpetas vacías...")
    for item in Config.BASE_DIR.iterdir():
        if item.is_dir() and item.name not in ['train', 'test', 'lost+found']:
            # Intentar borrar (solo borrará si está realmente vacía)
            subprocess.run(f"find {item} -type d -empty -delete", shell=True)
            if not any(item.iterdir()):
                item.rmdir()

    print(f"\n✨ ¡Organización 70/30 completada!")
    print(f"   - Train: {Config.TRAIN_DIR}")
    print(f"   - Test:  {Config.TEST_DIR}")

if __name__ == "__main__":
    import subprocess
    main()