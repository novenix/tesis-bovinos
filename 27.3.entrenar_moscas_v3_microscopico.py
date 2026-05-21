# COMANDO PARA EJECUTAR (CUANDO TERMINE EL 27.2):
# nohup bash -c "/opt/anaconda3/envs/tesis_vacas/bin/python -u /data/estudiantes/vacas/27.3.entrenar_moscas_v3_microscopico.py" > logs/output_v27_v3_micro.log 2>&1 &

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
            f.write(f"\n--- INICIO MONITOREO V27.3 MICRO {datetime.now()} ---\n")
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
    run_name = "yolo26_moscas_v3_micro_1024"
    
    # Logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "telemetria_v27_v3_micro.log"
    watcher = ResourceWatcher(log_path)
    watcher.start()

    print(f"🚀 Preparando Entrenamiento V3: Enfoque Microscópico")
    print(f"🎯 Estrategia: Grayscale, Single Class, High Box Gain, Warmup++")

    # Mantenemos el modelo Medium para aprovechar la capacidad con resolución alta
    model = YOLO("yolo26m-seg.pt")

    # Entrenamiento con configuraciones ultra-especializadas
    model.train(
        data=dataset_yaml,
        epochs=150,
        batch=12,            
        imgsz=1024,          
        device="2,3,4,5,6,7",
        project=project_name,
        name=run_name,
        exist_ok=True,
        
        # PUNTO 2: Clasificación Binaria Estricta
        single_cls=True,     # El modelo solo aprende "mosca o nada"
        
        # PUNTO 3: Ajuste de Ganancia de Pérdida (Extremo)
        box=20.0,            # Triple del peso original para precisión quirúrgica en localización
        cls=1.5,             # Aumento de peso en clasificación
        dfl=2.5,             # Refinamiento de bordes de bbox
        
        # PUNTO 4: Entrenamiento en Escala de Grises (Filtros de color en 0)
        hsv_h=0.0,           # Desactivar cambios de tono
        hsv_s=0.0,           # Forzar escala de grises en aumentos (Enfoque en textura)
        hsv_v=0.4,           # Mantener varianza de brillo
        
        # PUNTO 5: Warmup Agresivo y Learning Rate
        warmup_epochs=10,    # 10 épocas de "calentamiento" para estabilizar pesos
        lr0=0.0001,          # Empezar con un aprendizaje más fino/lento
        cos_lr=True,         # Descenso suave
        
        # Configuraciones de Estabilidad
        freeze=10,           # Seguimos protegiendo el backbone de COCO
        mosaic=1.0,          
        mixup=0.2,           
        retina_masks=True,   
        optimizer='AdamW',
        plots=True,
        save=True,
        workers=4
    )

if __name__ == "__main__":
    main()
