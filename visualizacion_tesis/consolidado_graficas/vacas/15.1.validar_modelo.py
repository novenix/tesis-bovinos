import os
# --- CONFIGURACIÓN DE SEGURIDAD PARA CRATOS (DEBE IR AL PRINCIPIO) ---
# Ocultamos la GPU 0 que está ocupada por otro usuario
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

import sys
import psutil
import time
import threading
import gc
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
import torch

# --- CLASE VIGILANTE CON LIMPIEZA ACTIVA (Adaptada del Script 15) ---
class ResourceWatcher(threading.Thread):
    def __init__(self, log_path, interval=15):
        super().__init__()
        self.log_path = log_path
        self.interval = interval
        self.daemon = True
        self.stop_event = threading.Event()

    def run(self):
        with open(self.log_path, "a") as f:
            f.write(f"\n--- INICIO MONITOREO VALIDACIÓN V15.1 {datetime.now()} ---\n")
            f.write("Timestamp | System_Used(GB) | Free(GB) | Proc_RAM(GB)\n")
            process = psutil.Process(os.getpid())
            while not self.stop_event.is_set():
                try:
                    # Limpieza forzada de basura en cada ciclo de monitoreo
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    now = datetime.now().strftime("%H:%M:%S")
                    mem = psutil.virtual_memory()
                    proc_mem = process.memory_info().rss / (1024**3)
                    for child in process.children(recursive=True):
                        try: proc_mem += child.memory_info().rss / (1024**3)
                        except: pass
                    
                    f.write(f"{now} | {mem.used/(1024**3):.2f} | {mem.available/(1024**3):.2f} | {proc_mem:.2f}\n")
                    f.flush()
                except: pass
                time.sleep(self.interval)

def main():
    # 1. Definir rutas (basadas en el entrenamiento del script 15)
    training_run_path = "runs/segment/tesis_bovinos_seg/entrenamiento_nano_puro_epoch1"
    model_path = f"{training_run_path}/weights/best.pt"
    dataset_yaml = "/data/estudiantes/vacas/dataset_final_yolo/dataset.yaml"
    
    # Los resultados se guardarán en: training_run_path / validacion_final_v15
    run_name = "validacion_final_v15"

    # Iniciar Vigilante de Recursos
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "telemetria_v15.1_validacion.log"
    watcher = ResourceWatcher(log_path)
    watcher.start()

    try:
        # 2. Cargar el mejor modelo guardado
        print(f"📂 Cargando el mejor modelo de: {model_path}")
        model = YOLO(model_path)

        # 3. Ejecutar la validación formal
        print("🚀 Iniciando validación formal en el conjunto de 'val'...")
        results = model.val(
            data=dataset_yaml,
            batch=4,           # Reducido aún más para máxima estabilidad
            imgsz=640,
            device="0,1,2,3",  # OJO: Como ocultamos la GPU 0 física, la 1 física ahora es la 0 lógica
            project=training_run_path, 
            name=run_name,             
            plots=True,       
            save_json=True    
        )

        print(f"✅ Validación completada. Resultados guardados en: {training_run_path}/{run_name}")
        
        # Imprimir resumen de métricas en consola
        print("\n--- RESUMEN DE MÉTRICAS (BEST MODEL) ---")
        print(f"mAP50 (Box): {results.box.map50:.4f}")
        print(f"mAP50-95 (Box): {results.box.map:.4f}")
        print(f"mAP50 (Mask): {results.seg.map50:.4f}")
        print(f"mAP50-95 (Mask): {results.seg.map:.4f}")

    finally:
        watcher.stop_event.set()
        print("🏁 Proceso de validación finalizado.")

if __name__ == "__main__":
    main()
