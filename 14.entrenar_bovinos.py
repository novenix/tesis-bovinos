import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import psutil
import time
import threading
from datetime import datetime

# --- CLASE VIGILANTE DE RECURSOS (TELEMETRÍA) ---
class ResourceWatcher(threading.Thread):
    def __init__(self, log_path, interval=5):
        super().__init__()
        self.log_path = log_path
        self.interval = interval
        self.daemon = True
        self.stop_event = threading.Event()

    def run(self):
        with open(self.log_path, "a") as f:
            f.write(f"\n--- INICIO MONITOREO {datetime.now()} ---\n")
            f.write("Timestamp | CPU% | System_RAM_Used(GB) | System_RAM_Free(GB) | Process_RAM(GB)\n")
            
            process = psutil.Process(os.getpid())
            
            while not self.stop_event.is_set():
                try:
                    now = datetime.now().strftime("%H:%M:%S")
                    cpu = psutil.cpu_percent()
                    mem = psutil.virtual_memory()
                    # RAM del proceso actual y sus hijos
                    proc_mem = process.memory_info().rss / (1024**3)
                    for child in process.children(recursive=True):
                        try:
                            proc_mem += child.memory_info().rss / (1024**3)
                        except: pass
                    
                    line = f"{now} | {cpu}% | {mem.used/(1024**3):.2f} | {mem.available/(1024**3):.2f} | {proc_mem:.2f}\n"
                    f.write(line)
                    f.flush()
                except Exception as e:
                    f.write(f"Error monitoreo: {e}\n")
                
                time.sleep(self.interval)

# --- CONFIGURACIÓN DE ARGUMENTOS ---
def main():
    parser = argparse.ArgumentParser(description="Entrenamiento YOLOv11 Multi-GPU con Telemetría")
    parser.add_argument("--gpu", type=str, default="1,2,3,4")
    parser.add_argument("--model", type=str, default="nano")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Iniciar Vigilante de RAM
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    watcher = ResourceWatcher(log_dir / "telemetria_recursos.log")
    watcher.start()

    # Rutas
    dataset_yaml = "/data/estudiantes/vacas/dataset_final_yolo/dataset.yaml"
    project_name = "tesis_bovinos_seg"
    run_name = "entrenamiento_nano_with_gridmask_multi_gpu"
    
    # Pesos para reanudar
    last_weights = Path(f"runs/segment/{project_name}/{run_name}/weights/last.pt")

    print(f"🚀 [DIAGNÓSTICO] Relanzando Nano en GPUs: {args.gpu}")
    
    # Cargar modelo (Reanudación forzada para el diagnóstico)
    if args.resume and last_weights.exists():
        print(f"🔄 Reanudando desde {last_weights}")
        model = YOLO(str(last_weights))
    else:
        model = YOLO("yolo11n-seg.pt")

    # CONFIGURACIÓN QUE FALLÓ (Para reproducir el error bajo vigilancia)
    train_params = {
        "data": dataset_yaml,
        "epochs": 100,
        "batch": 96,
        "workers": 16, # Este valor causó el pico sospechoso
        "imgsz": 640,
        "device": args.gpu,
        "project": project_name,
        "name": run_name,
        "exist_ok": True,
        "optimizer": "AdamW",
        "augment": True,
        "erasing": 0.5,
        "cache": False,
        "resume": args.resume
    }

    try:
        model.train(**train_params)
    finally:
        watcher.stop_event.set()
        print("🏁 Entrenamiento finalizado o interrumpido.")

if __name__ == "__main__":
    main()
