#!/usr/bin/env python3
"""
Script de Test Rápido - Un Solo Video (Versión Corregida)
"""

import sys
import os
from pathlib import Path
from pipeline_curaduria import process_single_video, Config

def main():
    if len(sys.argv) != 2:
        print("Uso: python test_single_video.py <ruta_al_video>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"❌ El archivo no existe: {video_path}")
        sys.exit(1)

    print("="*80)
    print(f"🧪 TEST DE VIDEO: {video_path.name}")
    print("="*80)

    # Forzar uso de GPU 0 para el test
    gpu_id = 0
    
    stats = process_single_video(video_path, gpu_id)

    print("\n" + "="*80)
    print("📊 RESULTADOS DEL TEST")
    print("="*80)
    print(f"Total frames extraídos: {stats['total_frames']}")
    print(f"Frames con ganado: {stats['frames_with_cattle']}")
    print(f"Frames guardados: {stats['frames_saved']}")
    
    if stats.get('status') == 'error':
        print(f"❌ ERROR: {stats.get('error')}")
    else:
        print(f"\n✅ Test finalizado. Revisa las imágenes en: {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()