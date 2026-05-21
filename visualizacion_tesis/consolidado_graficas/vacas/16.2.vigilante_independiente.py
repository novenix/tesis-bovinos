import os
import psutil
import time
from datetime import datetime
import torch
from pathlib import Path

def get_gpu_info():
    try:
        # Consulta temperatura y memoria de las 8 GPUs de Cratos
        info = os.popen("nvidia-smi --query-gpu=index,temperature.gpu,memory.used --format=csv,noheader,nounits").read().strip().split("
")
        return " | ".join(info)
    except:
        return "N/A"

def main():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "telemetria_v16_global.log"
    
    print(f"🕵️ Vigilante Independiente iniciado. Monitoreando en: {log_path}")
    
    with open(log_path, "a") as f:
        f.write(f"
--- INICIO MONITOREO GLOBAL V16 {datetime.now()} ---
")
        f.write("Timestamp | Sys_Used(GB) | Free(GB) | GPU_Index,Temp,Mem_Used...
")
        
        while True:
            try:
                now = datetime.now().strftime("%H:%M:%S")
                mem = psutil.virtual_memory()
                gpu_stats = get_gpu_info()
                
                log_line = f"{now} | {mem.used/(1024**3):.2f} | {mem.available/(1024**3):.2f} | {gpu_stats}
"
                f.write(log_line)
                f.flush()
                
                # Si la memoria libre baja de 5GB, lanzamos una alerta al log
                if mem.available / (1024**3) < 5.0:
                    f.write(f"⚠️ ALERTA: Memoria crítica detectada ({mem.available/(1024**3):.2f} GB)
")
                    f.flush()
                
            except Exception as e:
                print(f"Error en vigilancia: {e}")
            
            time.sleep(15)

if __name__ == "__main__":
    main()
