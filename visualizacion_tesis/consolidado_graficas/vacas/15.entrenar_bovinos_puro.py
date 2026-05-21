import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import psutil
import time
import threading
import gc
from datetime import datetime

# --- CLASE VIGILANTE CON LIMPIEZA ACTIVA (COPIA FIEL DEL 14) ---
class ResourceWatcher(threading.Thread):
    def __init__(self, log_path, interval=15):
        super().__init__()
        self.log_path = log_path
        self.interval = interval
        self.daemon = True
        self.stop_event = threading.Event()

    def run(self):
        with open(self.log_path, "a") as f:
            f.write(f"\n--- INICIO MONITOREO COLA INTELIGENTE V15 {datetime.now()} ---\n")
            f.write("Timestamp | System_Used(GB) | Free(GB) | Proc_RAM(GB)\n")
            process = psutil.Process(os.getpid())
            while not self.stop_event.is_set():
                try:
                    # Limpieza forzada de basura en cada ciclo de monitoreo
                    gc.collect()
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
    # Parámetros del experimento
    dataset_yaml = "/data/estudiantes/vacas/dataset_final_yolo/dataset.yaml"
    project_name = "tesis_bovinos_seg"
    run_name = "entrenamiento_nano_puro_epoch1"
    
    # Iniciar Vigilante
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "telemetria_v15_puro.log"
    watcher = ResourceWatcher(log_path)
    watcher.start()

    print("🚀 Lanzando Entrenamiento PURO (Época 1) con COLA INTELIGENTE...")
    
    # Cargar pesos originales (NADA de resume)
    model = YOLO("yolo11n-seg.pt")

    # CONFIGURACIÓN DE COLA INTELIGENTE (Copia fiel del 14 funcional)
    train_params = {
        "data": dataset_yaml,
        "epochs": 100,
        "batch": 64,         # 16 por GPU
        "workers": 2,        # SOLO 2 POR GPU (Cola Inteligente)
        "imgsz": 640,
        "device": "1,2,3,4",
        "project": project_name,
        "name": run_name,
        "exist_ok": True,
        "optimizer": "AdamW",
        "augment": True,
        "erasing": 0.5,
        "cache": False,
        "plots": True,
        "save": True,
        "close_mosaic": 10
    }

    try:
        model.train(**train_params)
    finally:
        watcher.stop_event.set()
        print("🏁 Entrenamiento finalizado o interrumpido.")

if __name__ == "__main__":
    main()
