# 🐄 Pipeline de Curaduría Automatizada - Detección de Ganado

> **Tesis de Maestría**: Sistema de extracción inteligente de frames con detección semántica usando YOLO-World v2

## 📋 Descripción

Pipeline automatizado para procesar videos DVR (.dav) de cámaras de seguridad, extrayendo únicamente frames con presencia de ganado bovino mediante detección open-vocabulary.

### 🎯 Características

- ✅ **Extracción inteligente**: Muestreo de 1 frame cada 3 segundos (configurable)
- ✅ **Detección semántica**: YOLO-World v2 con prompts de texto ("cow", "cattle", "bull")
- ✅ **Ahorro de espacio**: Solo guarda frames con ganado detectado (conf > 0.4)
- ✅ **Sin permisos sudo**: Instalación completa via Anaconda/pip
- ✅ **Logs detallados**: Reportes JSON y CSV con estadísticas de detección
- ✅ **Robusto**: Manejo de errores, timeouts, y archivos corruptos

---

## 🚀 Instalación Rápida

### Paso 1: Crear entorno Anaconda

```bash
# Activar conda (si no está activo)
source ~/anaconda3/bin/activate  # O la ruta donde instalaste Anaconda

# Crear entorno dedicado
conda create -n tesis_vacas python=3.10 -y
conda activate tesis_vacas
```

### Paso 2: Instalar ffmpeg (sin sudo)

```bash
# Desde conda-forge (repositorio con binarios precompilados)
conda install -c conda-forge ffmpeg -y
```

### Paso 3: Instalar paquetes de Python

```bash
# Librería principal con YOLO-World
pip install ultralytics==8.1.24

# Dependencias de procesamiento
pip install opencv-python-headless  # OpenCV sin GUI
pip install tqdm pandas Pillow      # Utilidades
```

### Paso 4: Verificar instalación

```bash
cd /data/estudiantes/vacas
python verificar_entorno.py
```

Si todo está ✅, continúa al siguiente paso.

---

## 📂 Estructura de Directorios

```
/data/estudiantes/vacas/
├── datosCrudos/              # Videos .dav originales (INPUT)
│   ├── camara1_20240101.dav
│   ├── camara2_20240101.dav
│   └── ...
├── dataset_curado/           # Frames con ganado (OUTPUT)
│   ├── camara1_20240101_frame0042.jpg
│   └── ...
├── frames_temp/              # Temporal (se limpia automáticamente)
├── logs/                     # Reportes JSON/CSV
│   ├── curaduria_pipeline_20260204_143022.json
│   └── curaduria_pipeline_20260204_143022.csv
├── pipeline_curaduria.py     # Script principal ⭐
├── verificar_entorno.py      # Script de verificación
└── README.md                 # Este archivo
```

---

## ⚙️ Configuración

Edita la clase `Config` en [pipeline_curaduria.py](pipeline_curaduria.py:30) para ajustar parámetros:

```python
class Config:
    # Directorios (ajustar si cambian rutas)
    INPUT_DIR = Path("/data/estudiantes/vacas/datosCrudos")
    OUTPUT_DIR = Path("/data/estudiantes/vacas/dataset_curado")

    # Extracción de frames
    FRAME_INTERVAL_SECONDS = 3  # ⬅️ Cambiar para más/menos frames

    # Detección
    MODEL_NAME = "yolov8l-worldv2.pt"  # l=Large (preciso), m=Medium (rápido)
    CONFIDENCE_THRESHOLD = 0.4         # ⬅️ Subir para más precisión (0.5-0.6)
                                       #    Bajar para más recall (0.3)
    TARGET_CLASSES = ["cow", "cattle", "bull", "calf", "livestock"]

    # Hardware
    USE_GPU = True  # ⬅️ Cambiar a False si no hay GPU/CUDA
```

### 🎛️ Recomendaciones de Configuración

| Escenario | `FRAME_INTERVAL` | `CONFIDENCE_THRESHOLD` | `MODEL_NAME` |
|-----------|------------------|------------------------|--------------|
| **Alta precisión** (pocos falsos positivos) | 3s | 0.5-0.6 | yolov8l-worldv2 |
| **Balanceado** (default) | 3s | 0.4 | yolov8l-worldv2 |
| **Alta cobertura** (no perder vacas) | 2s | 0.3 | yolov8x-worldv2 |
| **Rápido** (prototipo) | 5s | 0.4 | yolov8m-worldv2 |

---

## 🎬 Ejecución

### Modo Normal (Recomendado)

```bash
# Activar entorno
conda activate tesis_vacas

# Ejecutar pipeline
cd /data/estudiantes/vacas
python pipeline_curaduria.py
```

### Modo Background (para sesiones largas)

```bash
# Ejecutar en segundo plano con logs
nohup python pipeline_curaduria.py > pipeline_output.log 2>&1 &

# Ver progreso en tiempo real
tail -f pipeline_output.log

# Obtener PID del proceso
ps aux | grep pipeline_curaduria

# Detener si es necesario
kill <PID>
```

### Ejecución en Subconjunto (Testing)

Para probar con pocos videos primero:

```bash
# Crear carpeta de prueba
mkdir -p /data/estudiantes/vacas/datosCrudos_test

# Copiar 2-3 videos de ejemplo
cp /data/estudiantes/vacas/datosCrudos/video1.dav datosCrudos_test/

# Editar pipeline_curaduria.py línea 37:
# INPUT_DIR = Path("/data/estudiantes/vacas/datosCrudos_test")

# Ejecutar
python pipeline_curaduria.py
```

---

## 📊 Salidas y Resultados

### 1. Imágenes Curadas

```
dataset_curado/
├── camara1_20240101_20260204_143022_0042.jpg
├── camara1_20240101_20260204_143105_0087.jpg
└── ...
```

**Formato de nombres**: `{video_original}_{timestamp_procesamiento}_{frame_numero}.jpg`

### 2. Logs JSON (Detallado)

[logs/curaduria_pipeline_20260204_143022.json](logs/)

```json
[
  {
    "video_name": "camara1_20240101",
    "total_frames": 120,
    "frames_with_cattle": 45,
    "frames_saved": 45,
    "detections": [
      {
        "frame": "camara1_20240101_20260204_143022_0042.jpg",
        "class": "cow",
        "confidence": 0.87,
        "bbox": [123.4, 567.8, 890.1, 1234.5]
      }
    ]
  }
]
```

### 3. Logs CSV (Resumen)

[logs/curaduria_pipeline_20260204_143022.csv](logs/)

| video | total_frames | frames_con_ganado | frames_guardados | tasa_deteccion |
|-------|--------------|-------------------|------------------|----------------|
| camara1_20240101 | 120 | 45 | 45 | 37.50% |
| camara2_20240101 | 240 | 12 | 12 | 5.00% |

---

## 🐛 Troubleshooting

### ❌ Error: "ModuleNotFoundError: No module named 'ultralytics'"

**Solución**:
```bash
conda activate tesis_vacas  # Asegúrate de estar en el entorno correcto
pip install ultralytics
```

---

### ❌ Error: "ffmpeg: command not found"

**Solución**:
```bash
# Opción 1: Instalar con conda
conda install -c conda-forge ffmpeg -y

# Opción 2: Verificar que esté en PATH
which ffmpeg  # Debe mostrar una ruta

# Opción 3: Reinstalar en el entorno
conda activate tesis_vacas
conda install -c conda-forge ffmpeg --force-reinstall
```

---

### ❌ Error: "CUDA out of memory" o "RuntimeError: Cuda error"

**Solución**:
```python
# En pipeline_curaduria.py línea 50:
USE_GPU = False  # Cambiar a CPU

# O ajustar batch size (línea 48):
BATCH_SIZE = 1  # Procesar de uno en uno
```

---

### ⚠️ Warning: "No se encontraron archivos de video"

**Verificaciones**:
```bash
# 1. Verificar que existan archivos .dav
ls -lh /data/estudiantes/vacas/datosCrudos/*.dav

# 2. Verificar permisos de lectura
ls -la /data/estudiantes/vacas/datosCrudos/

# 3. Si están en subcarpetas, el pipeline los encontrará recursivamente
find /data/estudiantes/vacas/datosCrudos -name "*.dav"
```

---

### 🐢 Pipeline muy lento (< 1 frame/s)

**Optimizaciones**:

1. **Usar GPU** (10-50x más rápido):
   ```bash
   # Verificar GPU disponible
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Cambiar a modelo Medium** (2-3x más rápido):
   ```python
   MODEL_NAME = "yolov8m-worldv2.pt"  # En lugar de 'l'
   ```

3. **Aumentar intervalo de frames**:
   ```python
   FRAME_INTERVAL_SECONDS = 5  # En lugar de 3
   ```

---

### ❌ Error: "Permission denied" al escribir

**Solución**:
```bash
# Verificar permisos de escritura
ls -ld /data/estudiantes/vacas/dataset_curado

# Si no tienes permisos, solicita al administrador de Cratos:
chmod -R u+w /data/estudiantes/vacas/

# O cambia a tu directorio home:
# OUTPUT_DIR = Path("~/vacas_dataset_curado").expanduser()
```

---

## 🔬 Detalles Técnicos (para la Tesis)

### Arquitectura del Modelo

- **Base**: YOLOv8-World v2 (2024)
- **Tipo**: Open Vocabulary Object Detection (OVOD)
- **Backbone**: CSPDarknet53
- **Neck**: PANet
- **Head**: Grounding Head con text encoder CLIP
- **Parámetros**: 43.6M (Large variant)
- **Entrada**: 640x640 RGB (auto-resize)
- **Inferencia**: ~15ms/imagen (RTX 3090), ~200ms (CPU)

### Comparación con SOTA

| Método | Año | mAP@0.5 | Vocabulario | FPS (GPU) |
|--------|-----|---------|-------------|-----------|
| YOLO-World v2 | 2024 | 49.8 | Open | 65 |
| Grounding DINO | 2023 | 52.5 | Open | 12 |
| YOLOv8 (closed) | 2023 | 53.2 | Fixed 80 | 80 |

**Ventaja**: YOLO-World no requiere reentrenamiento para nuevas clases.

### Muestreo Temporal Justificación

- **1 frame/3s** → 1200 frames/hora → ~28,800 frames/día/cámara
- Videos típicos de seguridad: movimiento lento de ganado
- Redundancia temporal: vacas permanecen en campo de visión ~30-120s
- **Trade-off**: Balance entre cobertura y almacenamiento

### Métricas de Evaluación (para reportar)

```python
# De los logs JSON, calcular:
precision = frames_saved / frames_with_cattle  # Idealmente 1.0
recall = frames_with_cattle / total_frames_con_vacas_real  # Requiere anotación manual
f1_score = 2 * (precision * recall) / (precision + recall)
```

---

## 📚 Referencias

1. Cheng, T. et al. (2024). "YOLO-World: Real-Time Open-Vocabulary Object Detection". *CVPR 2024*.
2. Liu, S. et al. (2023). "Grounding DINO: Marrying DINO with Grounded Pre-Training". *ECCV 2023*.
3. Ultralytics (2024). "YOLOv8 Documentation". https://docs.ultralytics.com

---

## 📝 Notas Finales

### Para tu tesis:

1. **Sección de Metodología**: Citar YOLO-World, explicar arquitectura OVOD
2. **Validación**: Anotar manualmente 100-200 frames para calcular precision/recall
3. **Ablation Study**: Probar diferentes umbrales (0.3, 0.4, 0.5, 0.6) y reportar trade-offs
4. **Comparación**: Opcional - comparar con YOLOv8 estándar (requiere fine-tuning)

### Próximos pasos:

- [ ] Ejecutar pipeline completo
- [ ] Analizar logs y tasa de detección
- [ ] Revisar manualmente muestra de falsos positivos/negativos
- [ ] Ajustar `CONFIDENCE_THRESHOLD` si es necesario
- [ ] Generar dataset final para fase de conteo de moscas

---

## 💬 Soporte

Para dudas o problemas:

1. **Verificar logs**: `tail -n 50 logs/curaduria_pipeline_*.json`
2. **Modo verbose**: Editar `verbose=False` → `verbose=True` en línea 199
3. **Comunidad Ultralytics**: https://github.com/ultralytics/ultralytics/issues

---

**Autor**: [Tu Nombre]
**Institución**: [Tu Universidad]
**Fecha**: Febrero 2026
**Licencia**: MIT (uso académico)

---

*Generado con Claude Code - Anthropic AI*
