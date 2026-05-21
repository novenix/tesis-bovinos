
import os
import cv2
from ultralytics import YOLO
from pathlib import Path

# --- AISLAMIENTO DE GPU (USAMOS LA 1) ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    # 1. Configuración de Rutas
    model_path = "runs/segment/tesis_bovinos_yolo26/train2/weights/best.pt"
    val_images_dir = Path("dataset_v2_simplified/val/images")
    output_dir = Path("visualizacion_tesis/muestras_nano")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"❌ Error: No se encuentra el modelo en {model_path}")
        return

    # 2. Cargar Modelo con los mejores pesos actuales (Epoch 29 aprox)
    print(f"📂 Cargando pesos desde: {model_path}")
    model = YOLO(model_path)

    # 3. Seleccionar 5 imágenes de validación
    image_files = sorted([f for f in val_images_dir.glob("*.jpg")])[:5]
    
    print(f"🚀 Procesando {len(image_files)} muestras...")

    for i, img_path in enumerate(image_files):
        img_id = i + 1
        print(f"📸 Procesando Imagen {img_id}: {img_path.name}")

        # --- VERSIÓN 1: ORIGINAL ---
        img_orig = cv2.imread(str(img_path))
        cv2.imwrite(str(output_dir / f"img{img_id}_1_original.jpg"), img_orig)

        # --- VERSIÓN 2: REDUCIDA (640) ---
        img_640 = cv2.resize(img_orig, (640, 640))
        cv2.imwrite(str(output_dir / f"img{img_id}_2_640.jpg"), img_640)

        # --- VERSIÓN 3: SOLO MÁSCARAS (Segmentación Limpia) ---
        results = model.predict(source=str(img_path), imgsz=640, conf=0.25, verbose=False)[0]
        
        # Generar imagen con solo máscaras (boxes=False)
        res_masks = results.plot(boxes=False, labels=False, probs=False)
        cv2.imwrite(str(output_dir / f"img{img_id}_3_masks.jpg"), res_masks)

        # --- VERSIÓN 4: MÁSCARAS + BOUNDING BOXES ---
        res_full = results.plot(boxes=True, labels=True, probs=False)
        cv2.imwrite(str(output_dir / f"img{img_id}_4_full.jpg"), res_full)

    print(f"\n✅ Proceso completado. Revisa la carpeta: {output_dir}")

if __name__ == "__main__":
    main()
