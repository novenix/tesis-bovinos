# COMANDO PARA EJECUTAR:
# nohup bash -c "/opt/anaconda3/envs/tesis_vacas/bin/python -u /data/estudiantes/vacas/27.1.entrenar_moscas_v1.py" > logs/output_v27_manual.log 2>&1 &

import os
import threading
import time
import psutil
import gc
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
import torch

# --- PARCHE DE COMPATIBILIDAD 1080Ti ---
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

class ResourceWatcher(threading.Thread):
    def __init__(self, log_path, interval=15):
        super().__init__()
        self.log_path = log_path
        self.interval = interval
        self.daemon = True
        self.stop_event = threading.Event()

    def run(self):
        with open(self.log_path, "a") as f:
            f.write(f"\n--- INICIO MONITOREO V27.1 MANUAL {datetime.now()} ---\n")
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
    # Rutas del Dataset
    dataset_yaml = "/data/estudiantes/vacas/dataset_moscas_v2/final_dataset_moscas/train/segmented/data.yaml"
    project_name = "entrenamiento_manual_v27"
    run_name = "yolo26_moscas_v1_medium"
    
    # Logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "telemetria_v27_manual.log"
    watcher = ResourceWatcher(log_path)
    watcher.start()

    print(f"🚀 Iniciando Entrenamiento con Datos Manuales (YOLO 26 Medium)")
    print(f"📍 Dataset: {dataset_yaml}")

    # Cargar Pesos Medium
    model = YOLO("yolo26m-seg.pt")

    # Configuración de Entrenamiento
    model.train(
        data=dataset_yaml,
        epochs=100,
        batch=48,            # Ajustado para 6 GPUs (8 por GPU)
        imgsz=640,
        device="2,3,4,5,6,7", # Usando GPUs 2 a 7 como solicitado
        project=project_name,
        name=run_name,
        exist_ok=True,
        retina_masks=True,
        optimizer='AdamW',
        plots=True,
        save=True,
        workers=8
    )

if __name__ == "__main__":
    main()
