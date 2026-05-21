#!/usr/bin/env python3
import os
import torch
from ultralytics import YOLO
from pathlib import Path

# Imágenes que reportaste con personas
problem_images = [
    "2023-11-21_15.53.49-15.54.16[M][0@0][0]NADAA_f_0003.jpg",
    "2023-11-21_15.53.49-15.54.16[M][0@0][0]NADAA_f_0004.jpg",
    "2023-11-21_15.53.49-15.54.16[M][0@0][0]NADAA_f_0006.jpg",
    "2023-11-21_15.53.49-15.54.16[M][0@0][0]NADAA_f_0020.jpg",
    "2023-11-21_15.53.49-15.54.16[M][0@0][0]NADAA_f_0022.jpg",
    "2023-11-21_15.53.49-15.54.16[M][0@0][0]NADAA_f_0023.jpg",
    "2023-11-21_16.43.04-16.44.21[M][0@0][0]NADAA_f_0007.jpg",
    "2023-11-21_16.43.04-16.44.21[M][0@0][0]NADAA_f_0091.jpg"
]

IMG_DIR = Path("/data/estudiantes/vacas/dataset_curado/train/images")
thresholds = [0.4, 0.5, 0.6, 0.7]
classes = ["cow", "cattle", "bull", "calf", "livestock"]

print("🧪 TEST 5.1: PROBANDO UMBRALES DE CONFIANZA")
print("="*60)

model = YOLO("yolov8l-worldv2.pt")
model.set_classes(classes)

for img_name in problem_images:
    img_path = IMG_DIR / img_name
    if not img_path.exists(): 
        print(f"⚠️ No se encontró: {img_name}")
        continue
    
    print(f"\n🖼️ Imagen: {img_name}")
    for conf in thresholds:
        results = model.predict(source=str(img_path), conf=conf, verbose=False)
        count = len(results[0].boxes)
        status = "❌ DETECTADO" if count > 0 else "✅ LIMPIO"
        print(f"   Conf {conf}: {status} ({count} bultos)")