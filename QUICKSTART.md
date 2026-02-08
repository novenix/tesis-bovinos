# 🚀 Guía de Inicio Rápido - 5 Minutos

## TL;DR - Comandos Esenciales

```bash
# 1. Instalar todo automáticamente
cd /data/estudiantes/vacas
bash setup_entorno.sh

# 2. Activar entorno (CADA VEZ que abras terminal)
conda activate tesis_vacas

# 3. Verificar que todo esté OK
python verificar_entorno.py

# 4. Test con un video (recomendado primero)
python test_single_video.py datosCrudos/tu_primer_video.dav

# 5. Si todo OK, procesar todo el dataset
python pipeline_curaduria.py

# 6. Ver resultados
ls -lh dataset_curado/
cat logs/*.csv
```

---

## ⚡ Instalación (Primera Vez)

### Opción A: Automática (Recomendada)

```bash
cd /data/estudiantes/vacas
bash setup_entorno.sh
```

✅ Esto instala todo en ~5-10 minutos.

### Opción B: Manual (Si tienes problemas)

```bash
# Crear entorno
conda create -n tesis_vacas python=3.10 -y
conda activate tesis_vacas

# Instalar ffmpeg
conda install -c conda-forge ffmpeg -y

# Instalar Python packages
pip install ultralytics==8.1.24 opencv-python-headless tqdm pandas

# Verificar
python verificar_entorno.py
```

---

## 📂 Preparar Tus Datos

```bash
# Copiar videos .dav al directorio correcto
cp /ruta/origen/*.dav /data/estudiantes/vacas/datosCrudos/

# Verificar que se copiaron
ls -lh /data/estudiantes/vacas/datosCrudos/

# Ver cuántos videos tienes
ls datosCrudos/*.dav | wc -l
```

---

## 🧪 Test Rápido (Antes de Procesar Todo)

```bash
# Activar entorno
conda activate tesis_vacas

# Listar videos disponibles
ls datosCrudos/*.dav

# Testear con el primero
python test_single_video.py datosCrudos/[nombre_primer_video].dav

# Revisar resultados del test
ls -lh dataset_curado/*test*
```

### ¿Qué esperar del test?

✅ **Éxito**: Ves mensajes como:
```
✓ Extraídos 120 frames
✓ Frames con ganado: 45/120 (37.5%)
✓ Frames guardados: 45
```

❌ **Problema**: Si dice "0 frames con ganado":
- Baja el umbral: Edita `pipeline_curaduria.py` línea 44: `CONFIDENCE_THRESHOLD = 0.3`
- Verifica que el video tenga vacas visibles

---

## 🎬 Ejecución Principal

### Modo Normal (Terminal Activo)

```bash
conda activate tesis_vacas
cd /data/estudiantes/vacas
python pipeline_curaduria.py
```

**Ventaja**: Ves progreso en tiempo real
**Desventaja**: Debes mantener terminal abierta

### Modo Background (Recomendado para Datasets Grandes)

```bash
conda activate tesis_vacas
cd /data/estudiantes/vacas

# Ejecutar en segundo plano
nohup python pipeline_curaduria.py > logs/pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Ver progreso en tiempo real
tail -f logs/pipeline_*.log

# O monitorear número de frames guardados
watch -n 10 'ls dataset_curado/*.jpg | wc -l'
```

**Para detener**:
```bash
ps aux | grep pipeline_curaduria
kill [PID]
```

---

## 📊 Ver Resultados

### Imágenes Curadas

```bash
# Contar frames guardados
ls dataset_curado/*.jpg | wc -l

# Ver los primeros
ls dataset_curado/*.jpg | head -n 5

# Tamaño total del dataset
du -sh dataset_curado/

# Copiar muestra a tu máquina local (con scp)
scp usuario@cratos:/data/estudiantes/vacas/dataset_curado/*.jpg ~/Desktop/muestra/
```

### Logs y Estadísticas

```bash
# Ver resumen CSV (más legible)
cat logs/*.csv

# Ver log JSON completo (todas las detecciones)
cat logs/*.json | jq '.'  # Si tienes jq instalado

# Calcular estadísticas globales
python -c "
import pandas as pd
df = pd.read_csv('logs/[tu_log].csv')
print(f'Total videos: {len(df)}')
print(f'Total frames: {df.total_frames.sum()}')
print(f'Frames con ganado: {df.frames_con_ganado.sum()}')
print(f'Tasa promedio: {df.frames_con_ganado.sum()/df.total_frames.sum()*100:.2f}%')
"
```

---

## ⚙️ Ajustar Parámetros (Configuración)

Edita [pipeline_curaduria.py](pipeline_curaduria.py:30-54):

### Parámetros Más Comunes

```python
# Línea 40: Cuánto espacio entre frames (segundos)
FRAME_INTERVAL_SECONDS = 3  # ⬅️ Cambiar a 2 para más frames, 5 para menos

# Línea 44: Qué tan seguro debe estar el modelo (0-1)
CONFIDENCE_THRESHOLD = 0.4  # ⬅️ Bajar a 0.3 para más detecciones
                            #    Subir a 0.5 para menos falsos positivos

# Línea 43: Modelo (tamaño vs velocidad)
MODEL_NAME = "yolov8l-worldv2.pt"  # l=Large (default)
# MODEL_NAME = "yolov8m-worldv2.pt"  # m=Medium (2x más rápido)
# MODEL_NAME = "yolov8x-worldv2.pt"  # x=Extra Large (más preciso, más lento)

# Línea 50: Usar GPU o CPU
USE_GPU = True  # ⬅️ Cambiar a False si no tienes GPU
```

Después de cambiar, volver a ejecutar:
```bash
python pipeline_curaduria.py
```

---

## 🐛 Problemas Comunes

### ❌ "conda: command not found"

```bash
# Activar conda manualmente
source ~/anaconda3/bin/activate
# O
source ~/miniconda3/bin/activate

# Luego
conda activate tesis_vacas
```

### ❌ "ffmpeg: command not found"

```bash
conda activate tesis_vacas
conda install -c conda-forge ffmpeg -y
```

### ❌ "CUDA out of memory"

```bash
# Editar pipeline_curaduria.py línea 50
USE_GPU = False

# O línea 48
BATCH_SIZE = 1
```

### ❌ "No se encontraron archivos de video"

```bash
# Verificar que estén en el lugar correcto
ls /data/estudiantes/vacas/datosCrudos/

# Si están en otra ubicación, copiarlos
cp /ruta/real/*.dav /data/estudiantes/vacas/datosCrudos/
```

### ❌ "Permission denied"

```bash
# Verificar permisos
ls -ld /data/estudiantes/vacas/

# Si no tienes permisos, contactar admin del servidor
# O cambiar directorio de salida a tu home
```

---

## 📚 Archivos de Documentación

| Archivo | Propósito |
|---------|-----------|
| **QUICKSTART.md** | Esta guía (inicio rápido) |
| [README.md](README.md) | Documentación completa |
| [CONFIGURACIONES_AVANZADAS.md](CONFIGURACIONES_AVANZADAS.md) | Casos de uso especiales |

---

## 🔄 Workflow Completo (Diagrama)

```
1. Instalación (una vez)
   └─> bash setup_entorno.sh

2. Cada sesión nueva
   └─> conda activate tesis_vacas

3. Preparar datos
   └─> cp videos.dav datosCrudos/

4. Test rápido
   └─> python test_single_video.py datosCrudos/video1.dav

5. Ajustar Config (si es necesario)
   └─> nano pipeline_curaduria.py  # Editar líneas 30-54

6. Ejecución completa
   └─> nohup python pipeline_curaduria.py > logs/run.log 2>&1 &

7. Monitorear progreso
   └─> tail -f logs/run.log

8. Ver resultados
   └─> ls dataset_curado/
   └─> cat logs/*.csv

9. Validación manual
   └─> Revisar muestra aleatoria de frames

10. Para tu tesis
    └─> Copiar dataset_curado/ y logs/ a tu máquina local
```

---

## ⏱️ Tiempos Estimados

| Tarea | GPU (RTX 3090) | CPU (16 cores) |
|-------|----------------|----------------|
| Test (1 video 30min) | ~2 minutos | ~10 minutos |
| 10 horas de video | ~30 minutos | ~3 horas |
| 100 horas de video | ~5 horas | ~30 horas |

**Variables que afectan velocidad**:
- Resolución del video (4K vs 720p)
- `FRAME_INTERVAL_SECONDS` (2s vs 5s)
- Modelo usado (Medium vs Extra Large)
- Formato del video (.dav encoding)

---

## 💡 Tips Pro

### 1. Procesar por Lotes

Si tienes 1000 videos, divide en lotes de 100:

```bash
# Crear carpetas temporales
mkdir datosCrudos_lote1 datosCrudos_lote2 ...

# Mover 100 videos a cada lote
mv datosCrudos/video_001-100.dav datosCrudos_lote1/

# Editar Config para procesar lote1
INPUT_DIR = Path("/data/estudiantes/vacas/datosCrudos_lote1")

# Procesar
python pipeline_curaduria.py

# Repetir para lote2, lote3...
```

### 2. Checkpoints Automáticos

El pipeline guarda progreso automáticamente. Si se interrumpe:

```bash
# Ver qué videos ya se procesaron
cat logs/*.json | grep video_name

# Mover videos ya procesados a carpeta "completados"
mkdir datosCrudos_completados
mv datosCrudos/video_procesado.dav datosCrudos_completados/

# Volver a ejecutar (procesará solo los que faltan)
python pipeline_curaduria.py
```

### 3. Monitoreo en Segundo Monitor/Pantalla

```bash
# Terminal 1: Pipeline
python pipeline_curaduria.py

# Terminal 2: GPU usage (si aplica)
watch -n 2 nvidia-smi

# Terminal 3: Frames guardados
watch -n 5 'ls dataset_curado/ | wc -l'

# Terminal 4: Espacio en disco
watch -n 10 'df -h /data/estudiantes/vacas'
```

---

## 📞 Ayuda Adicional

1. **Re-ejecutar verificación**: `python verificar_entorno.py`
2. **Ver logs recientes**: `tail -n 50 logs/*.log`
3. **Documentación completa**: `cat README.md`
4. **Configuraciones avanzadas**: `cat CONFIGURACIONES_AVANZADAS.md`

---

## ✅ Checklist de Inicio

- [ ] Setup ejecutado exitosamente
- [ ] `python verificar_entorno.py` muestra todo ✅
- [ ] Videos .dav copiados a `datosCrudos/`
- [ ] Test con un video completado
- [ ] Resultados del test revisados
- [ ] Parámetros ajustados (si es necesario)
- [ ] Pipeline completo ejecutándose
- [ ] Monitoreo configurado
- [ ] Primera revisión de resultados

---

**¿Listo? Ejecuta:**

```bash
conda activate tesis_vacas && python pipeline_curaduria.py
```

🐄 **¡Buena suerte con tu tesis!** 🐄
