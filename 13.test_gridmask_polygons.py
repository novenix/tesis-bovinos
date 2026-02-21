import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import random

def load_yolo_polygons(label_path):
    """Carga polígonos en formato relativo [class, x1, y1, x2, y2...]"""
    polygons = []
    if not label_path.exists(): return polygons
    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            points = np.array(parts[1:]).reshape(-1, 2)
            polygons.append((class_id, points))
    return polygons

def main():
    dataset_root = Path("/data/estudiantes/vacas/dataset_final_yolo/train")
    output_dir = Path("/data/estudiantes/vacas/verificacion_aumentos_full")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_files = list((dataset_root / "images").glob("*.jpg"))
    if not img_files:
        print("Error: No se encontraron imágenes.")
        return
    img_path = random.choice(img_files)
    
    img_orig = cv2.imread(str(img_path))
    label_path = dataset_root / "labels" / f"{img_path.stem}.txt"
    yolo_polygons = load_yolo_polygons(label_path)

    # Crear máscara para transformaciones sincronizadas
    mask = np.zeros(img_orig.shape[:2], dtype=np.uint8)
    for class_id, pts in yolo_polygons:
        h, w = img_orig.shape[:2]
        abs_pts = (pts * [w, h]).astype(np.int32)
        cv2.fillPoly(mask, [abs_pts], int(class_id) + 1)

    # 1. PIPELINE COMPLETO (AutoAugment Geométrico + Color + GridMask)
    pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=1.0, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(p=0.8),
        A.HueSaturationValue(p=0.8),
        A.GridDropout(ratio=0.5, unit_size_min=96, unit_size_max=224, random_offset=True, p=0.5)
    ])

    print(f"🧬 Generando secuencia GEOMÉTRICA para: {img_path.name}")
    
    # Variante 0: Original
    cv2.imwrite(str(output_dir / f"v0_original_{img_path.name}"), img_orig)

    for i in range(1, 6):
        transformed = pipeline(image=img_orig, mask=mask)
        img_aug = transformed['image']
        mask_aug = transformed['mask']
        
        res_img = img_aug.copy()
        for class_id in range(1, 5): # Clases 0, 1, 2, 3 mapeadas a 1, 2, 3, 4
            m = (mask_aug == class_id).astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = [(0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 255, 0)][class_id-1]
            
            # Dibujar contorno y relleno semitransparente
            cv2.drawContours(res_img, contours, -1, color, 2)
            overlay = res_img.copy()
            cv2.fillPoly(overlay, contours, color)
            cv2.addWeighted(overlay, 0.3, res_img, 0.7, 0, res_img)

        cv2.imwrite(str(output_dir / f"v{i}_augmented_{img_path.name}"), res_img)
        print(f"  ✅ v{i}: Variante con rotación/espejo/luz guardada.")

    print(f"\n🚀 Verificación GEOMÉTRICA lista en: {output_dir}")

if __name__ == "__main__":
    main()
