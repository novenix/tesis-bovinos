# COMANDO PARA EJECUTAR:
# /opt/anaconda3/envs/tesis_vacas/bin/python /data/estudiantes/vacas/27.3.validar_moscas_v2_dataset_pequeño.py

import os
from ultralytics import YOLO
from pathlib import Path

def main():
    # Rutas
    model_path = "/data/estudiantes/vacas/runs/segment/entrenamiento_manual_v27/yolo26_moscas_v2_pequeño_1024/weights/best.pt"
    dataset_yaml = "/data/estudiantes/vacas/dataset_moscas_v2/final_dataset_moscas/data_completo.yaml"
    project_name = "validacion_manual_v27"
    run_name = "val_moscas_v2_pequeño_1024"

    print(f"🚀 Iniciando VALIDACIÓN para Proyecto de Grado")
    print(f"📦 Modelo: {model_path}")
    print(f"🎯 Dataset: {dataset_yaml}")

    if not os.path.exists(model_path):
        print(f"❌ ERROR: No se encontró el modelo en {model_path}")
        return

    # Cargar el mejor modelo entrenado
    model = YOLO(model_path)

    # Ejecutar validación en el set de VAL
    results = model.val(
        data=dataset_yaml,
        split='val',         # Forzamos que use el set de validación
        imgsz=1024,
        device="2,3,4",      # Usando GPUs 2, 3 y 4
        project=project_name,
        name=run_name,
        exist_ok=True,
        plots=True,          # Generar matrices de confusión, curvas PR, etc.
        save_json=True,      # Guardar resultados para análisis detallado
        save_hybrid=True     # Guardar etiquetas e imágenes de referencia
    )

    print(f"✅ Validación completada. Resultados en: {project_name}/{run_name}")

if __name__ == "__main__":
    main()
