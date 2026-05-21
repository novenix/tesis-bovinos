
# COMANDO DE EJECUCIÓN (Segundo Plano):
# nohup /opt/anaconda3/envs/tesis_vacas/bin/python -u /data/estudiantes/vacas/16.4.preparar_test_rigor_nano.py > logs/output_v16_test_rigor.log 2>&1 &

import os
# --- CONFIGURACIÓN DE GPU (CRATOS 2) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
import subprocess
import shutil
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from ultralytics import YOLO

# --- CONFIGURACIÓN DE RUTAS ---
BASE_PATH = Path("/data/estudiantes/vacas")
VIDEO_TEST_DIR = BASE_PATH / "dataset_curado/test/videos"
TEMP_FRAMES_DIR = BASE_PATH / "frames_temp/test_rigor_nano"
OUTPUT_DATASET_DIR = BASE_PATH / "dataset_v2_simplified/test"
SAM3_PATH = BASE_PATH / "sam3"
BEST_MODEL_PATH = BASE_PATH / "runs/segment/tesis_bovinos_yolo26/train2/weights/best.pt"

# --- CONFIGURACIÓN IA ---
YOLO_FILTER_MODEL = "yolov8l-worldv2.pt"
CONF_FILTER = 0.4
TARGET_CLASSES = ["cow", "cattle", "bull", "calf", "livestock", "person"]
CATTLE_CLASSES = ["cow", "cattle", "bull", "calf", "livestock"]
SIMPLIFY_TARGET = 100 # Para Nano

# Agregar SAM3 al path
sys.path.append(str(SAM3_PATH))
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def simplify_polygon(coords, target_points=100):
    pts = np.array(coords).reshape(-1, 2)
    low, high = 0.0, 0.01
    best_epsilon = 0.001
    for _ in range(10):
        mid = (low + high) / 2
        epsilon = mid * cv2.arcLength(pts.astype(np.float32), True)
        approx = cv2.approxPolyDP(pts.astype(np.float32), epsilon, True)
        if len(approx) > target_points: low = mid
        else: high = mid; best_epsilon = mid
    final_epsilon = best_epsilon * cv2.arcLength(pts.astype(np.float32), True)
    approx = cv2.approxPolyDP(pts.astype(np.float32), final_epsilon, True)
    return approx.reshape(-1).tolist()

def main():
    print("🚀 [NANO] Iniciando Preparación de Test de Rigor (Pipeline Espejo)...")
    
    # 1. Crear directorios
    (OUTPUT_DATASET_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DATASET_DIR / "labels").mkdir(parents=True, exist_ok=True)
    TEMP_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Cargar Modelos
    print("🤖 Cargando YOLO Filter...")
    yolo_filter = YOLO(YOLO_FILTER_MODEL)
    yolo_filter.set_classes(TARGET_CLASSES)
    print("🤖 Cargando SAM 3...")
    sam3_model = build_sam3_image_model(checkpoint_path=str(SAM3_PATH / "checkpoints/sam3.pt"), device="cuda", load_from_HF=False)
    print("🤖 Modelos Cargados.")
    sam3_model.eval()
    processor = Sam3Processor(sam3_model, confidence_threshold=0.3)

    # 3. Identificar Videos de Test
    import random
    video_files = list(VIDEO_TEST_DIR.rglob("*.dav")) + list(VIDEO_TEST_DIR.rglob("*.mp4"))
    random.seed(42)
    if len(video_files) > 100:
        video_files = random.sample(video_files, 100)
    print(f"📹 Seleccionados {len(video_files)} videos de test aleatorios.")

    class_map = {"cow": 0, "head": 1, "leg": 2, "tail": 3}

    for v_path in tqdm(video_files, desc="Procesando Videos"):
        v_stem = f"{v_path.parent.name}_{v_path.stem}".replace(' ', '_')
        video_temp = TEMP_FRAMES_DIR / v_stem
        if video_temp.exists(): shutil.rmtree(video_temp)
        video_temp.mkdir(parents=True, exist_ok=True)
        cmd = ['ffmpeg', '-i', str(v_path), '-vf', 'fps=1/10', '-qscale:v', '2', '-y', str(video_temp / "f_%04d.jpg")]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        frames = sorted(video_temp.glob("*.jpg"))
        for frame_path in frames:
            # A. Filtrar con YOLO (Espejo del pipeline de train)
            results = yolo_filter.predict(source=str(frame_path), conf=CONF_FILTER, verbose=False)
            if not results or len(results[0].boxes) == 0:
                frame_path.unlink(); continue
            
            keep = False
            for box in results[0].boxes:
                cls_name = TARGET_CLASSES[int(box.cls[0])]
                if cls_name in CATTLE_CLASSES: keep = True
                if cls_name == "person" and float(box.conf[0]) > 0.5: keep = False; break
            
            if not keep:
                frame_path.unlink(); continue

            # B. Etiquetar con SAM 3
            if not frame_path.exists(): continue
            
            img_pil = Image.open(frame_path).convert("RGB")
            w, h = img_pil.size
            label_name = f"test_{v_stem}_{frame_path.stem}.txt"
            label_path = OUTPUT_DATASET_DIR / "labels" / label_name
            
            state = processor.set_image(img_pil)
            label_created = False
            
            # Full Cow
            state = processor.set_text_prompt("cow", state)
            if "boxes" in state and len(state["boxes"]) > 0:
                for i in range(len(state["boxes"])):
                    label_created = True
                    # ...
                    mask = state["masks"][i, 0].cpu().numpy()
                    mask_np = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        if len(cnt) < 3: continue
                        # Correct way to flatten for simplify_polygon
                        pts_flat = []
                        for p in cnt: pts_flat.extend([p[0][0]/w, p[0][1]/h])
                        
                        simplified = simplify_polygon(pts_flat, SIMPLIFY_TARGET)
                        with open(label_path, 'a') as f:
                            f.write(f"0 " + " ".join([f"{x:.6f}" for x in simplified]) + "\n")

            # Key Parts
            for part in ["head", "leg", "tail"]:
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(f"cow {part}", state)
                if "boxes" in state and len(state["boxes"]) > 0:
                    for i in range(len(state["boxes"])):
                        mask = state["masks"][i, 0].cpu().numpy()
                        mask_np = (mask * 255).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if len(cnt) < 3: continue
                            pts_flat = []
                            for p in cnt: pts_flat.extend([p[0][0]/w, p[0][1]/h])
                            simplified = simplify_polygon(pts_flat, SIMPLIFY_TARGET)
                            with open(label_path, 'a') as f:
                                f.write(f"{class_map[part]} " + " ".join([f"{x:.6f}" for x in simplified]) + "\n")

            # Mover imagen final solo si se creó etiqueta
            if label_created:
                shutil.move(str(frame_path), OUTPUT_DATASET_DIR / "images" / f"test_{v_stem}_{frame_path.name}")
            else:
                if frame_path.exists(): frame_path.unlink()

        if video_temp.exists():
            shutil.rmtree(video_temp)

    # 4. Actualizar YAML para incluir test
    yaml_path = BASE_PATH / "dataset_v2_simplified/dataset_v2.yaml"
    with open(yaml_path, 'r') as f:
        import yaml
        cfg = yaml.safe_load(f)
    
    cfg['test'] = 'test/images'
    with open(yaml_path, 'w') as f:
        yaml.dump(cfg, f)

    print(f"✅ [NANO] Test Rigor preparado en: {OUTPUT_DATASET_DIR}")
    
    # 5. Ejecutar Validación Final de Test
    print("📊 Iniciando Evaluación Final en Test Set...")
    model = YOLO(BEST_MODEL_PATH)
    model.val(data=str(yaml_path), split='test', imgsz=640, project="runs/segment/tesis_bovinos_yolo26/train2", name="test_rigor_nano")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open("logs/crash_nano.log", "w") as f:
            import traceback
            f.write(str(e) + "\n")
            f.write(traceback.format_exc())
        print(f"❌ CRASH: {e}")
        sys.exit(1)
