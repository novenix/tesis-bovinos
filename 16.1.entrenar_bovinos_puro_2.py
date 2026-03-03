# COMANDO EJECUTADO EN SESIÓN (MARZO 01, 2026):
# nohup bash -c "/opt/anaconda3/envs/tesis_vacas/bin/python -u /data/estudiantes/vacas/16.1.entrenar_bovinos_puro_2.py" > logs/output_v16_train.log 2>&1 &

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

# --- CLASE VIGILANTE (IDÉNTICA AL SCRIPT 15) ---
class ResourceWatcher(threading.Thread):
    def __init__(self, log_path, interval=15):
        super().__init__()
        self.log_path = log_path
        self.interval = interval
        self.daemon = True
        self.stop_event = threading.Event()

    def run(self):
        with open(self.log_path, "a") as f:
            f.write(f"\n--- INICIO MONITOREO V16 {datetime.now()} ---\n")
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
    dataset_yaml = "/data/estudiantes/vacas/dataset_v2_simplified/dataset_v2.yaml"
    project_name = "tesis_bovinos_yolo26"
    run_name = "train2"
    
    if os.environ.get("RANK", "0") == "0":
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "telemetria_v16_yolo26.log"
        watcher = ResourceWatcher(log_path)
        watcher.start()

    print("🚀 Lanzando Entrenamiento V16 (YOLO26) con Gestión Interna de Logs...")
    
    model = YOLO("yolo26n-seg.pt")

    model.train(
        data=dataset_yaml,
        epochs=100,
        batch=80,
        imgsz=640,
        device="3,4,5,6,7",
        project=project_name,
        name=run_name,
        exist_ok=True,
        optimizer="MuSGD",
        freeze=10,
        box=25.0, # En Ultralytics YOLO, 'box' controla tanto Box como Mask loss.
        cls=0.5,
        augment=True,
        erasing=0.15,
        auto_augment="randaugment",
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,
        plots=True,
        save=True,
        cache=False,
        workers=2
    )

if __name__ == "__main__":
    main()
