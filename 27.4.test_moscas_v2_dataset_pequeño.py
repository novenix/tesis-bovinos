# COMANDO PARA EJECUTAR:
# /opt/anaconda3/envs/tesis_vacas/bin/python /data/estudiantes/vacas/27.4.test_moscas_v2_dataset_pequeño.py

import os
from ultralytics import YOLO
from pathlib import Path

def main():
    # Rutas
    model_path = "/data/estudiantes/vacas/runs/segment/entrenamiento_manual_v27/yolo26_moscas_v2_pequeño_1024/weights/best.pt"
    dataset_yaml = "/data/estudiantes/vacas/dataset_moscas_v2/final_dataset_moscas/data_completo.yaml"
    project_name = "test_manual_v27"
    run_name = "test_moscas_v2_pequeño_1024"

    print(f"🚀 Iniciando TEST FINAL para Proyecto de Grado")
    print(f"📦 Modelo: {model_path}")
    print(f"🎯 Dataset: {dataset_yaml}")

    if not os.path.exists(model_path):
        print(f"❌ ERROR: No se encontró el modelo en {model_path}")
        return

    # Cargar el mejor modelo entrenado
    model = YOLO(model_path)

    # Ejecutar validación en el set de TEST
    results = model.val(
        data=dataset_yaml,
        split='test',        # AQUÍ ESTÁ LA CLAVE: Usamos el set de TEST
        imgsz=1024,
        device="5,6,7",      # Usando GPUs 5, 6 y 7
        project=project_name,
        name=run_name,
        exist_ok=True,
        plots=True,          # Generar matrices de confusión y curvas para la tesis
        save_json=True,
        save_hybrid=True     # Guardar etiquetas e imágenes de referencia
    )

    print(f"✅ Test completado. Métricas finales guardadas en: {project_name}/{run_name}")

if __name__ == "__main__":
    main()
