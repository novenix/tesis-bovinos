# COMANDO PARA EJECUTAR:
# nohup bash -c "/opt/anaconda3/envs/tesis_vacas/bin/python -u /data/estudiantes/vacas/27.2.entrenar_moscas_v2_dataset_pequeño.py" > logs/output_v27_pequeño.log 2>&1 &

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
            f.write(f"\n--- INICIO MONITOREO V27.2 PEQUEÑO {datetime.now()} ---\n")
            process = psutil.Process(os.getpid())
            while not self.stop_event.is_set():
                try:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    now = datetime.now().strftime("%H:%M:%S")
                    mem = psutil.virtual_memory()
                    proc_mem = process.memory_info().rss / (1024**3)
                    gpu_info = os.popen("nvidia-smi --query-gpu=index,temperature.gpu,memory.used --format=csv,noheader,nounits").read().replace("\n", " | ").strip()
                    f.write(f"{now} | Sys:{mem.used/(1024**3):.2f}G | Proc:{proc_mem:.2f}G | GPU:{gpu_info}\n")
                    f.flush()
                except: pass
                time.sleep(self.interval)

def main():
    # Rutas
    dataset_yaml = "/data/estudiantes/vacas/dataset_moscas_v2/final_dataset_moscas/train/segmented/data.yaml"
    project_name = "entrenamiento_manual_v27"
    run_name = "yolo26_moscas_v2_pequeño_1024"
    
    # Logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "telemetria_v27_pequeño.log"
    watcher = ResourceWatcher(log_path)
    watcher.start()

    print(f"🚀 Iniciando Entrenamiento Optimizado para Dataset Pequeño (YOLO 26 Medium)")
    print(f"🎯 Estrategia: imgsz=1024, Freeze=10, High Loss Gain")

    model = YOLO("yolo26m-seg.pt")

    # Entrenamiento con configuraciones extremas (Puntos 1, 3, 4 y 5)
    model.train(
        data=dataset_yaml,
        epochs=150,          # Subimos un poco para compensar el Freeze
        batch=12,            # Reducido para aguantar 1024px en 11GB de VRAM (2 por GPU)
        imgsz=1024,          # PUNTO 1: Alta resolución para detalles
        device="2,3,4,5,6,7",# Usando 6 GPUs
        project=project_name,
        name=run_name,
        exist_ok=True,
        
        # PUNTO 3: Tiling/Mosaic (Simulado con mosaic y crop alto)
        mosaic=1.0,          
        mixup=0.2,           
        crop_fraction=1.0,   
        
        # PUNTO 4: Congelar Backbone (Capas iniciales)
        freeze=10,           
        
        # PUNTO 5: Ajuste de Ganancia de Pérdida (Castigo fuerte por errores)
        box=15.0,            # Default es 7.5 (Doblamos el peso de la caja)
        cls=1.0,             # Default es 0.5 (Doblamos el peso de la clase)
        dfl=2.0,             # Default es 1.5
        
        # Otras mejoras
        retina_masks=True,   
        optimizer='AdamW',
        lr0=0.001,           # Aprendizaje más estable
        cos_lr=True,         # Curva de aprendizaje suave
        plots=True,
        save=True,
        workers=4
    )

if __name__ == "__main__":
    main()
