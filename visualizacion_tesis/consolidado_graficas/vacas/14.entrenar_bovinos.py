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

# --- CLASE VIGILANTE CON LIMPIEZA ACTIVA ---
class ResourceWatcher(threading.Thread):
    def __init__(self, log_path, interval=15):
        super().__init__()
        self.log_path = log_path
        self.interval = interval
        self.daemon = True
        self.stop_event = threading.Event()

    def run(self):
        with open(self.log_path, "a") as f:
            f.write(f"\n--- MONITOREO COLA INTELIGENTE {datetime.now()} ---\n")
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
    parser = argparse.ArgumentParser(description="Entrenamiento YOLOv11 - COLA INTELIGENTE")
    parser.add_argument("--gpu", type=str, default="1,2,3,4")
    parser.add_argument("--model", type=str, default="nano")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Iniciar Vigilante
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    watcher = ResourceWatcher(log_dir / "telemetria_cola_inteligente.log")
    watcher.start()

    dataset_yaml = "/data/estudiantes/vacas/dataset_final_yolo/dataset.yaml"
    project_name = "tesis_bovinos_seg"
    run_name = "entrenamiento_nano_with_gridmask_multi_gpu"
    
    last_weights = Path(f"runs/segment/{project_name}/{run_name}/weights/last.pt")

    if args.resume and last_weights.exists():
        print(f"🔄 Reanudando con COLA INTELIGENTE desde {last_weights}")
        model = YOLO(str(last_weights))
    else:
        model = YOLO("yolo11n-seg.pt")

    # CONFIGURACIÓN DE COLA INTELIGENTE (Mínima RAM)
    train_params = {
        "data": dataset_yaml,
        "epochs": 100,
        "batch": 64,         # 16 por GPU para evitar picos de VRAM
        "workers": 2,        # SOLO 2 POR GPU. Esto es la "cola corta" que pediste.
        "imgsz": 640,
        "device": args.gpu,
        "project": project_name,
        "name": run_name,
        "exist_ok": True,
        "optimizer": "AdamW",
        "augment": True,
        "erasing": 0.5,
        "cache": False,      # Prohibido cachear en RAM fotos de 160GB
        "resume": args.resume,
        "plots": True,
        "overlap_mask": True,
        "close_mosaic": 10
    }

    try:
        model.train(**train_params)
    finally:
        watcher.stop_event.set()

if __name__ == "__main__":
    main()
