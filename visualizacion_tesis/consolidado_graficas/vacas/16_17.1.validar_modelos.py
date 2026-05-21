
import os
import sys
import psutil
import time
import threading
import gc
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
import torch

# --- CONFIGURACIÓN DE SEGURIDAD PARA CRATOS ---
# Usamos estrictamente las GPUs 2, 3, 4 y 5 como solicitó el usuario
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"

class ResourceWatcher(threading.Thread):
    def __init__(self, log_path, interval=15):
        super().__init__()
        self.log_path = log_path
        self.interval = interval
        self.daemon = True
        self.stop_event = threading.Event()

    def run(self):
        with open(self.log_path, "a") as f:
            f.write(f"\n--- INICIO MONITOREO VALIDACIÓN V16/V17 {datetime.now()} ---\n")
            f.write("Timestamp | System_Used(GB) | Free(GB) | Proc_RAM(GB)\n")
            process = psutil.Process(os.getpid())
            while not self.stop_event.is_set():
                try:
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

def validar_modelo(model_path, dataset_yaml, project_path, run_name, log_file):
    print(f"\n🔍 VALIDANDO: {run_name}")
    print(f"📂 Modelo: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: No se encuentra el modelo en {model_path}")
        return

    model = YOLO(model_path)
    
    # Reducimos batch a 4 para evitar OOM con retina_masks en GPUs de 11GB
    results = model.val(
        data=dataset_yaml,
        batch=4,          
        imgsz=640,
        device="0,1,2,3",  # Índices lógicos mapeados a las GPUs físicas 2,3,4,5
        project=project_path,
        name=run_name,
        plots=True,
        save_json=True
    )
    
    with open(log_file, "a") as f:
        f.write(f"\n--- RESULTADOS FINAL {run_name} ---\n")
        f.write(f"mAP50 (Box): {results.box.map50:.4f}\n")
        f.write(f"mAP50-95 (Box): {results.box.map:.4f}\n")
        f.write(f"mAP50 (Mask): {results.seg.map50:.4f}\n")
        f.write(f"mAP50-95 (Mask): {results.seg.map:.4f}\n")
    
    print(f"✅ Validación {run_name} completada.")

def main():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    telemetry_log = log_dir / "telemetria_validacion_v16_v17.log"
    results_summary = log_dir / "resumen_validaciones_finales.txt"
    
    if results_summary.exists():
        results_summary.unlink()

    watcher = ResourceWatcher(telemetry_log)
    watcher.start()

    try:
        # --- VALIDACIÓN V16 (NANO) ---
        validar_modelo(
            model_path="runs/segment/tesis_bovinos_yolo26/train2/weights/best.pt",
            dataset_yaml="dataset_v2_simplified/dataset_v2.yaml",
            project_path="runs/segment/tesis_bovinos_yolo26/train2",
            run_name="validacion_final_v16",
            log_file=results_summary
        )

        # --- VALIDACIÓN V17 (MEDIUM) ---
        validar_modelo(
            model_path="runs/segment/tesis_bovinos_medium/train_v17_medium/weights/best.pt",
            dataset_yaml="dataset_v3_medium/dataset_v3.yaml",
            project_path="runs/segment/tesis_bovinos_medium/train_v17_medium",
            run_name="validacion_final_v17",
            log_file=results_summary
        )

    finally:
        watcher.stop_event.set()
        print(f"\n🏁 Proceso finalizado. Resumen en: {results_summary}")

if __name__ == "__main__":
    main()
