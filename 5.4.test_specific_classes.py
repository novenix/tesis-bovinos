#!/usr/bin/env python3
import os
import torch
from ultralytics import YOLO
from pathlib import Path

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
classes = ["cow", "cattle"]

print("🧪 TEST 5.3: PROBANDO CLASES ESPECÍFICAS (SIN LIVESTOCK)")
print("="*60)

model = YOLO("yolov8l-worldv2.pt")
model.set_classes(classes)

for img_name in problem_images:
    img_path = IMG_DIR / img_name
    if not img_path.exists(): continue
    
    results = model.predict(source=str(img_path), conf=0.4, verbose=False)
    print(f"\n🖼️ Imagen: {img_name}")
    
    if len(results[0].boxes) == 0:
        print("   ✅ LIMPIO (No detectó nada)")
    else:
        for box in results[0].boxes:
            cls_name = classes[int(box.cls[0])]
            conf = float(box.conf[0])
            print(f"   ❌ Detectado como: {cls_name} (Conf: {conf:.2f})")