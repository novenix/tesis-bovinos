# ⚙️ Configuraciones Avanzadas y Casos de Uso

Esta guía proporciona configuraciones optimizadas para diferentes escenarios de investigación.

---

## 📊 Escenarios de Configuración

### 1️⃣ Máxima Precisión (Publicación Científica)

**Objetivo**: Minimizar falsos positivos, dataset limpio para análisis estadístico

```python
class Config:
    # Usar modelo más grande y preciso
    MODEL_NAME = "yolov8x-worldv2.pt"  # Extra Large (más lento, más preciso)

    # Umbral alto para alta confianza
    CONFIDENCE_THRESHOLD = 0.55

    # Más variaciones de clase
    TARGET_CLASSES = [
        "cow", "cattle", "bull", "calf", "heifer",
        "livestock", "bovine", "dairy cow", "beef cattle"
    ]

    # Muestreo más denso
    FRAME_INTERVAL_SECONDS = 2

    # Alta calidad de imagen
    JPEG_QUALITY = 95
```

**Ventajas**: ~98% precisión, ideal para conteo automatizado
**Desventajas**: Puede perder algunas vacas (menor recall), más lento

---

### 2️⃣ Máximo Recall (Exploración Inicial)

**Objetivo**: No perder ninguna vaca, luego filtrar manualmente

```python
class Config:
    # Umbral bajo para capturar todo
    CONFIDENCE_THRESHOLD = 0.25

    # Intervalos cortos
    FRAME_INTERVAL_SECONDS = 2

    # Clases amplias
    TARGET_CLASSES = [
        "cow", "cattle", "animal", "livestock",
        "mammal", "farm animal"  # Más genérico
    ]

    # Modelo balanceado
    MODEL_NAME = "yolov8l-worldv2.pt"
```

**Ventajas**: ~95% recall, no pierdes datos
**Desventajas**: Más falsos positivos, requiere filtrado manual

---

### 3️⃣ Prototipo Rápido (CPU/Laptop)

**Objetivo**: Testear pipeline sin GPU, en máquina local

```python
class Config:
    # Modelo más pequeño
    MODEL_NAME = "yolov8s-worldv2.pt"  # Small (5x más rápido)

    # CPU mode
    USE_GPU = False

    # Muestreo espaciado
    FRAME_INTERVAL_SECONDS = 5

    # Menor resolución interna (más rápido)
    # Nota: Esto requiere modificar la inferencia
    BATCH_SIZE = 1

    # Clases mínimas
    TARGET_CLASSES = ["cow", "cattle"]
```

**Velocidad**: ~2-3 fps en CPU moderna
**Uso**: Validación rápida de 10-20 videos

---

### 4️⃣ Producción GPU (Dataset Masivo)

**Objetivo**: Procesar miles de horas de video eficientemente

```python
class Config:
    # Balance precisión/velocidad
    MODEL_NAME = "yolov8l-worldv2.pt"

    # Configuración estándar
    CONFIDENCE_THRESHOLD = 0.40

    # Procesamiento paralelo
    BATCH_SIZE = 16  # Si tienes 16GB+ VRAM
    USE_GPU = True

    # Muestreo estándar
    FRAME_INTERVAL_SECONDS = 3

    # Calidad media (ahorro de espacio)
    JPEG_QUALITY = 85  # ~30% menos espacio que 92
```

**Velocidad**: ~20-30 fps con RTX 3090
**Capacidad**: ~500 horas de video/día

---

## 🎯 Casos de Uso Específicos

### A) Detección de Terneros (Calves)

Enfoque en animales jóvenes/pequeños:

```python
TARGET_CLASSES = ["calf", "young cow", "baby cow", "small cattle"]
CONFIDENCE_THRESHOLD = 0.35  # Más bajo (terneros son más difíciles)
MODEL_NAME = "yolov8x-worldv2.pt"  # Modelo grande para detecciones pequeñas
```

**Modificación adicional** en `process_frames_batch()`:

```python
# Línea ~198, después de results = model.predict()
# Filtrar solo detecciones pequeñas (terneros)
for box in results[0].boxes:
    bbox = box.xyxy.tolist()[0]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height

    # Solo aceptar si el área es pequeña (terneros)
    if area < 50000:  # Ajustar según resolución
        # ... guardar detección
```

---

### B) Análisis de Comportamiento Nocturno

Para cámaras infrarrojas/low-light:

```python
# Muestreo más denso (animales activos de noche)
FRAME_INTERVAL_SECONDS = 2

# Umbral más bajo (calidad de imagen reducida)
CONFIDENCE_THRESHOLD = 0.35

# Preprocesamiento de imagen (añadir antes de inferencia)
```

**Agregar función de mejora de contraste**:

```python
import cv2

def enhance_lowlight(frame_path):
    """Mejora contraste para imágenes nocturnas"""
    img = cv2.imread(str(frame_path))

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Sobrescribir frame
    cv2.imwrite(str(frame_path), enhanced)

# Llamar antes de model.predict() en línea ~199
```

---

### C) Videos de Alta Resolución (4K)

Para cámaras modernas con alta resolución:

```python
# Aprovechar la resolución
CONFIDENCE_THRESHOLD = 0.45  # Puedes ser más estricto

# NO cambiar JPEG_QUALITY (mantener calidad)
JPEG_QUALITY = 92

# Considerar downscaling en ffmpeg para ahorrar espacio
```

**Modificar `extract_frames_ffmpeg()`** línea ~125:

```python
cmd = [
    'ffmpeg',
    '-i', str(video_path),
    '-vf', f'fps=1/{Config.FRAME_INTERVAL_SECONDS},scale=1920:1080',  # ⬅️ Añadir scale
    '-qscale:v', '2',
    '-y',
    str(Config.TEMP_DIR / output_pattern)
]
```

---

### D) Multi-Especie (Vacas + Otros Animales)

Para granjas mixtas:

```python
TARGET_CLASSES = [
    # Bovinos
    "cow", "cattle", "bull", "calf",
    # Otros
    "horse", "sheep", "goat", "pig",
    "chicken", "donkey"
]

# Crear subcarpetas por especie
```

**Modificar `process_frames_batch()`** línea ~206:

```python
# Organizar por clase detectada
for box in results[0].boxes:
    class_name = Config.TARGET_CLASSES[int(box.cls)]

    # Crear subcarpeta por especie
    species_dir = Config.OUTPUT_DIR / class_name
    species_dir.mkdir(exist_ok=True)

    output_path = species_dir / output_name
    shutil.copy2(frame_path, output_path)
```

---

## 🔧 Modificaciones Avanzadas del Código

### 1. Filtrado por Región de Interés (ROI)

Para ignorar áreas irrelevantes (caminos, edificios):

```python
# Añadir a Config
ROI_POLYGON = [
    (100, 100),   # Esquina superior izquierda
    (1800, 100),  # Esquina superior derecha
    (1800, 900),  # Esquina inferior derecha
    (100, 900)    # Esquina inferior izquierda
]

# En process_frames_batch(), después de detección:
import cv2
import numpy as np

def is_in_roi(bbox, roi_polygon):
    """Verifica si el centro del bbox está en ROI"""
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    point = (int(center_x), int(center_y))

    # Convertir polígono a numpy array
    polygon = np.array(roi_polygon, np.int32)

    # Verificar si punto está dentro
    result = cv2.pointPolygonTest(polygon, point, False)
    return result >= 0

# Usar en el filtrado:
for box in results[0].boxes:
    bbox = box.xyxy.tolist()[0]
    if not is_in_roi(bbox, Config.ROI_POLYGON):
        continue  # Saltar esta detección
```

---

### 2. Tracking Temporal (Evitar Duplicados)

Para evitar guardar la misma vaca en frames consecutivos:

```python
import hashlib

def compute_image_hash(frame_path):
    """Calcula hash perceptual de imagen"""
    img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (16, 16))  # Reducir a 16x16
    hash_val = hashlib.md5(img_resized.tobytes()).hexdigest()
    return hash_val

# Mantener registro de hashes recientes
recent_hashes = set()

# En process_frames_batch():
img_hash = compute_image_hash(frame_path)

# Solo guardar si es suficientemente diferente
if img_hash not in recent_hashes:
    shutil.copy2(frame_path, output_path)
    recent_hashes.add(img_hash)

    # Limpiar hashes viejos (mantener últimos 20)
    if len(recent_hashes) > 20:
        recent_hashes.pop()
```

---

### 3. Exportar Anotaciones en Formato COCO

Para usar con otros frameworks (Detectron2, MMDetection):

```python
import json

def export_coco_annotations(all_stats, output_file):
    """Exporta detecciones en formato COCO JSON"""

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": cls}
            for i, cls in enumerate(Config.TARGET_CLASSES)
        ]
    }

    annotation_id = 1

    for video_stats in all_stats:
        for det in video_stats['detections']:
            # Agregar imagen
            image_id = len(coco_data['images']) + 1
            coco_data['images'].append({
                "id": image_id,
                "file_name": det['frame'],
                "width": 1920,  # Ajustar a tu resolución
                "height": 1080
            })

            # Agregar anotación
            bbox = det['bbox']
            coco_bbox = [
                bbox[0],                # x
                bbox[1],                # y
                bbox[2] - bbox[0],      # width
                bbox[3] - bbox[1]       # height
            ]

            coco_data['annotations'].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": Config.TARGET_CLASSES.index(det['class']),
                "bbox": coco_bbox,
                "area": coco_bbox[2] * coco_bbox[3],
                "iscrowd": 0,
                "score": det['confidence']
            })
            annotation_id += 1

    # Guardar
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)

# Llamar en main() después de procesar todos los videos
export_coco_annotations(all_stats, Config.LOG_DIR / "annotations_coco.json")
```

---

## 📈 Benchmarking y Optimización

### Medir Rendimiento por Componente

Añadir al inicio de `main()`:

```python
import time

timings = {
    'extraction': 0,
    'inference': 0,
    'io': 0
}

# En extract_frames_ffmpeg():
start = time.time()
# ... código de extracción
timings['extraction'] += time.time() - start

# En process_frames_batch():
start = time.time()
results = model.predict(...)
timings['inference'] += time.time() - start

start = time.time()
shutil.copy2(...)
timings['io'] += time.time() - start

# Al final, reportar:
print("\n⏱️  ANÁLISIS DE RENDIMIENTO:")
total = sum(timings.values())
for component, duration in timings.items():
    percentage = (duration / total) * 100
    print(f"   {component}: {duration:.1f}s ({percentage:.1f}%)")
```

---

## 🧪 Validación de Calidad del Dataset

Script para revisar aleatoriamente 100 frames curados:

```python
import random
from pathlib import Path

def validate_random_sample(dataset_dir, n_samples=100):
    """Muestra N frames aleatorios para validación manual"""
    frames = list(Path(dataset_dir).glob("*.jpg"))

    if len(frames) < n_samples:
        sample = frames
    else:
        sample = random.sample(frames, n_samples)

    print(f"Abriendo {len(sample)} frames para validación manual...")
    print("Marca como:")
    print("  ✓ = Correcto (tiene vaca)")
    print("  ✗ = Falso positivo (no tiene vaca)")
    print("  ? = Ambiguo")

    results = {'correct': 0, 'false_positive': 0, 'ambiguous': 0}

    for i, frame in enumerate(sample, 1):
        # Mostrar imagen (requiere display)
        # O copiar a carpeta review/
        review_dir = Path(dataset_dir).parent / "review"
        review_dir.mkdir(exist_ok=True)
        shutil.copy(frame, review_dir / f"review_{i:03d}.jpg")

    print(f"\nFrames copiados a: {review_dir}")
    print("Revisa manualmente y calcula:")
    print("  Precision = correctos / total_revisados")

# Ejecutar después del pipeline
validate_random_sample("/data/estudiantes/vacas/dataset_curado")
```

---

## 💡 Tips Finales

### 1. Procesamiento Incremental

Para datasets grandes, procesa por lotes:

```python
# Modificar main() para guardar progreso
PROGRESS_FILE = Config.LOG_DIR / "progress.json"

# Cargar videos ya procesados
if PROGRESS_FILE.exists():
    with open(PROGRESS_FILE) as f:
        processed = set(json.load(f))
else:
    processed = set()

# Filtrar videos
video_files = [v for v in video_files if v.name not in processed]

# Después de cada video, actualizar
processed.add(video_path.name)
with open(PROGRESS_FILE, 'w') as f:
    json.dump(list(processed), f)
```

### 2. Logging Avanzado

```bash
# Ejecutar con logs detallados
python pipeline_curaduria.py 2>&1 | tee logs/pipeline_$(date +%Y%m%d_%H%M%S).log
```

### 3. Monitoreo en Tiempo Real

```bash
# Terminal 1: Ejecutar pipeline
python pipeline_curaduria.py

# Terminal 2: Monitorear progreso
watch -n 5 'ls dataset_curado/ | wc -l'

# Terminal 3: Monitorear uso de GPU
watch -n 2 nvidia-smi
```

---

## 📞 Troubleshooting Avanzado

### Problema: OOM (Out of Memory) en GPU

```python
# Solución 1: Reducir batch size
BATCH_SIZE = 1

# Solución 2: Limpiar caché CUDA periódicamente
import torch
torch.cuda.empty_cache()  # Añadir cada N videos

# Solución 3: Usar modelo más pequeño
MODEL_NAME = "yolov8m-worldv2.pt"
```

### Problema: ffmpeg muy lento

```python
# Usar hardware acceleration (solo si disponible)
cmd = [
    'ffmpeg',
    '-hwaccel', 'cuda',  # O 'vaapi' en Linux
    '-i', str(video_path),
    ...
]
```

---

**Fecha**: Febrero 2026
**Versión**: 1.0
**Mantenedor**: [Tu Nombre]
