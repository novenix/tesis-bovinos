import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Añadir el directorio del repositorio clonado al path
sys.path.append("/data/estudiantes/vacas/sam3")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def main():
    # Configuración de rutas
    checkpoint = "/data/estudiantes/vacas/sam3/checkpoints/sam3.pt"
    image_dir = Path("/data/estudiantes/vacas/dataset_curado/train/images")
    output_dir = Path("/data/estudiantes/vacas/verificacion_etiquetas")
    output_dir.mkdir(exist_ok=True)

    # Cargar modelo
    print("Cargando modelo SAM 3...")
    model = build_sam3_image_model(
        checkpoint_path=checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_from_HF=False # Ya los bajamos manualmente
    )
    model.eval()

    # Inicializar procesador
    processor = Sam3Processor(model, confidence_threshold=0.3)

    # Seleccionar 5 imágenes para prueba
    image_paths = sorted(list(image_dir.glob("*.jpg")))[:5]
    
    # Prompts a probar
    prompts = ["cow", "person"]

    for img_path in image_paths:
        print(f"\nProcesando {img_path.name}...")
        image = Image.open(img_path).convert("RGB")
        
        # Copia de la imagen para dibujar
        img_cv = cv2.imread(str(img_path))
        
        # 1. Detectar Vacas
        print("  Buscando 'cow'...")
        state = processor.set_image(image)
        state = processor.set_text_prompt("cow", state)
        
        if "boxes" in state and len(state["boxes"]) > 0:
            boxes = state["boxes"].cpu().numpy()
            scores = state["scores"].cpu().numpy()
            for box, score in zip(boxes, scores):
                cv2.rectangle(img_cv, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(img_cv, f"cow {score:.2f}", (int(box[0]), int(box[1])-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"  Encontradas {len(boxes)} vacas.")
        else:
            print("  No se encontraron vacas.")

        # 2. Detectar Personas (Filtro Negativo)
        print("  Buscando 'person'...")
        # Reiniciamos prompts para la misma imagen
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt("person", state)
        
        if "boxes" in state and len(state["boxes"]) > 0:
            boxes = state["boxes"].cpu().numpy()
            scores = state["scores"].cpu().numpy()
            for box, score in zip(boxes, scores):
                cv2.rectangle(img_cv, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(img_cv, f"person {score:.2f}", (int(box[0]), int(box[1])-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(f"  Encontradas {len(boxes)} personas.")

        # Guardar imagen de verificación
        cv2.imwrite(str(output_dir / f"test_sam3_{img_path.name}"), img_cv)
        print(f"Resultado guardado en {output_dir / f'test_sam3_{img_path.name}'}")

if __name__ == "__main__":
    main()
