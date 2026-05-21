# COMANDO EJECUTADO EN SESIÓN (MARZO 05, 2026):
# nohup bash -c "/opt/anaconda3/envs/tesis_vacas/bin/python -u /data/estudiantes/vacas/17.1.entrenar_bovinos_medium.py" > logs/output_v17_medium.log 2>&1 &

import os
import sys
import threading
import time
import psutil
import gc
from datetime import datetime
from pathlib import Path

# --- PARCHE DE COMPATIBILIDAD 1080Ti ---
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import torch
from ultralytics import YOLO

# --- CLASE VIGILANTE (MANTENIDA POR ESTABILIDAD) ---
class ResourceWatcher(threading.Thread):
    def __init__(self, log_path, interval=15):
        super().__init__()
        self.log_path = log_path
        self.interval = interval
        self.daemon = True
        self.stop_event = threading.Event()

    def run(self):
        with open(self.log_path, "a") as f:
            f.write(f"\n--- INICIO MONITOREO V17 MEDIUM {datetime.now()} ---\n")
            f.write("Timestamp | Sys_Used(GB) | Free(GB) | Proc_RAM(GB) | GPU_Stats\n")
            process = psutil.Process(os.getpid())
            while not self.stop_event.is_set():
                try:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    now = datetime.now().strftime("%H:%M:%S")
                    mem = psutil.virtual_memory()
                    proc_mem = process.memory_info().rss / (1024**3)
                    
                    gpu_info = "N/A"
                    try:
                        gpu_info = os.popen("nvidia-smi --query-gpu=index,temperature.gpu,memory.used --format=csv,noheader,nounits").read().replace("\n", " | ").strip()
                    except: pass
                    
                    f.write(f"{now} | {mem.used/(1024**3):.2f} | {mem.available/(1024**3):.2f} | {proc_mem:.2f} | {gpu_info}\n")
                    f.flush()
                except: pass
                time.sleep(self.interval)

def main():
    # Dataset V3 (250 vértices)
    dataset_yaml = "/data/estudiantes/vacas/dataset_v3_medium/dataset_v3.yaml"
    project_name = "tesis_bovinos_medium"
    run_name = "train_v17_medium"
    
    if os.environ.get("RANK", "0") == "0":
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "telemetria_v17_medium.log"
        watcher = ResourceWatcher(log_path)
        watcher.start()

    print("🚀 Lanzando Entrenamiento V17 (YOLO26 Medium) con Retina Masks...")
    
    # Cargar Pesos Medium
    model = YOLO("yolo26m-seg.pt")

    # CONFIGURACIÓN OPTIMIZADA PARA MODELO MEDIUM
    model.train(
        data=dataset_yaml,
        epochs=100,
        batch=40,            # Reducido por tamaño de VRAM en Medium
        imgsz=640,
        device="3,4,5,6,7",  # GPUs solicitadas
        project=project_name,
        name=run_name,
        exist_ok=True,
        optimizer="MuSGD",
        freeze=10,
        box=25.0,            # Mantenemos prioridad de segmentación
        cls=0.5,
        augment=True,
        erasing=0.15,
        auto_augment="randaugment",
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,
        retina_masks=True,   # MEJORA CLAVE PARA MODELO MEDIUM
        plots=True,
        save=True,
        cache=False,
        workers=2
    )

if __name__ == "__main__":
    main()
