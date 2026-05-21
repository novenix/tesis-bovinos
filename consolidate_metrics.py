import os
import csv
import yaml
from pathlib import Path

def read_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except:
            return {}

def get_last_row_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        last_row = None
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            last_row = row
        return last_row

def list_visuals(directory):
    if not os.path.exists(directory):
        return []
    visuals = []
    extensions = ['.png', '.jpg', '.jpeg']
    try:
        for root, dirs, files in os.walk(directory):
            for f in files:
                if any(f.endswith(ext) for ext in extensions):
                    visuals.append(f)
            break # Top level only
    except:
        pass
    return sorted(list(set(visuals)))

def format_table(metrics_box, metrics_mask):
    return (
        "| Tipo | Precisión | Recall | mAP50 | mAP50-95 |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n"
        f"| **Box** | {metrics_box['p']} | {metrics_box['r']} | {metrics_box['mAP50']} | {metrics_box['mAP50-95']} |\n"
        f"| **Mask** | {metrics_mask['p']} | {metrics_mask['r']} | {metrics_mask['mAP50']} | {metrics_mask['mAP50-95']} |\n"
    )

def main():
    projects = [
        {
            "name": "Bovinos v16 (Nano)",
            "task": "Bovinos Segmentation",
            "train_dir": "runs/segment/tesis_bovinos_yolo26/train2",
            "val_dir": "runs/segment/runs/segment/tesis_bovinos_yolo26/train2/validacion_final_v16",
            "test_rigor_log": "logs/output_v16_test_rigor.log",
            "script_train": "16.1.entrenar_bovinos_puro_2.py",
            "script_val": "16_17.1.validar_modelos.py",
            "script_test": "16.4.preparar_test_rigor_nano.py",
            "val_metrics": {
                "box": {"p": "0.554", "r": "0.334", "mAP50": "0.370", "mAP50-95": "0.229"},
                "mask": {"p": "0.414", "r": "0.186", "mAP50": "0.186", "mAP50-95": "0.095"}
            },
            "test_rigor_metrics": {
                "box": {"p": "0.165", "r": "0.098", "mAP50": "0.084", "mAP50-95": "0.050"},
                "mask": {"p": "0.199", "r": "0.056", "mAP50": "0.046", "mAP50-95": "0.024"}
            }
        },
        {
            "name": "Bovinos v17 (Medium)",
            "task": "Bovinos Segmentation",
            "train_dir": "runs/segment/tesis_bovinos_medium/train_v17_medium",
            "val_dir": "runs/segment/runs/segment/tesis_bovinos_medium/train_v17_medium/validacion_final_v17",
            "test_rigor_log": "logs/output_v17_test_rigor.log",
            "script_train": "17.1.entrenar_bovinos_medium.py",
            "script_val": "16_17.1.validar_modelos.py",
            "script_test": "17.2.preparar_test_rigor_medium.py",
            "val_metrics": {
                "box": {"p": "0.672", "r": "0.429", "mAP50": "0.489", "mAP50-95": "0.344"},
                "mask": {"p": "0.461", "r": "0.253", "mAP50": "0.246", "mAP50-95": "0.125"}
            },
            "test_rigor_metrics": {
                "box": {"p": "0.290", "r": "0.168", "mAP50": "0.172", "mAP50-95": "0.114"},
                "mask": {"p": "0.179", "r": "0.101", "mAP50": "0.087", "mAP50-95": "0.040"}
            }
        },
        {
            "name": "Moscas v1 (Medium)",
            "task": "Moscas Segmentation",
            "train_dir": "runs/segment/entrenamiento_manual_v27/yolo26_moscas_v1_medium",
            "script_train": "27.1.entrenar_moscas_v1.py",
            "val_metrics": {
                "box": {"p": "0.341", "r": "0.335", "mAP50": "0.251", "mAP50-95": "0.069"},
                "mask": {"p": "0.336", "r": "0.322", "mAP50": "0.242", "mAP50-95": "0.068"}
            }
        },
        {
            "name": "Moscas v2 (Pequeño)",
            "task": "Moscas Segmentation",
            "train_dir": "runs/segment/entrenamiento_manual_v27/yolo26_moscas_v2_pequeño_1024",
            "val_dir": "runs/segment/validacion_manual_v27/val_moscas_v2_pequeño_1024",
            "test_dir": "runs/segment/test_manual_v27/test_moscas_v2_pequeño_1024",
            "script_train": "27.2.entrenar_moscas_v2",
            "script_test": "27.4.test_moscas_v2",
            "val_metrics": {
                "box": {"p": "0.537", "r": "0.467", "mAP50": "0.421", "mAP50-95": "0.136"},
                "mask": {"p": "0.489", "r": "0.489", "mAP50": "0.415", "mAP50-95": "0.109"}
            },
            "test_metrics": {
                "box": {"p": "0.425", "r": "0.487", "mAP50": "0.362", "mAP50-95": "0.102"},
                "mask": {"p": "0.384", "r": "0.457", "mAP50": "0.296", "mAP50-95": "0.069"}
            }
        },
        {
            "name": "Moscas v3 (Micro)",
            "task": "Moscas Segmentation",
            "train_dir": "runs/segment/entrenamiento_manual_v27/yolo26_moscas_v3_micro_1024",
            "script_train": "27.3.entrenar_moscas_v3",
            "val_metrics": {
                "box": {"p": "0.565", "r": "0.577", "mAP50": "0.530", "mAP50-95": "0.171"},
                "mask": {"p": "0.478", "r": "0.516", "mAP50": "0.426", "mAP50-95": "0.122"}
            }
        }
    ]

    report = "# 📊 DASHBOARD INTEGRAL DE MÉTRICAS - PROYECTO DE GRADO\n\n"
    report += "Este reporte consolida todos los resultados de entrenamiento, validación final y pruebas de rigor para Bovinos y Moscas.\n\n"

    for p in projects:
        report += f"## 🚀 {p['name']}\n"
        report += f"- **Objetivo:** {p['task']}\n"
        report += f"- **Directorio:** `{p['train_dir']}`\n"
        
        args = read_yaml(os.path.join(p['train_dir'], 'args.yaml'))
        if args:
            report += "### ⚙️ Configuración del Modelo\n"
            report += f"| Modelo Base | Tamaño Imagen | Batch | Epochs | Dataset |\n"
            report += f"| :--- | :--- | :--- | :--- | :--- |\n"
            report += f"| `{args.get('model')}` | `{args.get('imgsz')}` | `{args.get('batch')}` | `{args.get('epochs')}` | `{os.path.basename(str(args.get('data')))}` |\n\n"

        # 1. ENTRENAMIENTO (Last Epoch)
        results_csv = os.path.join(p['train_dir'], 'results.csv')
        last = get_last_row_csv(results_csv)
        if last:
            report += "### 📈 Resultados de Entrenamiento (Último Epoch)\n"
            m_box = {"p": last.get('metrics/precision(B)', 'N/A'), "r": last.get('metrics/recall(B)', 'N/A'), "mAP50": last.get('metrics/mAP50(B)', 'N/A'), "mAP50-95": last.get('metrics/mAP50-95(B)', 'N/A')}
            m_mask = {"p": last.get('metrics/precision(M)', 'N/A'), "r": last.get('metrics/recall(M)', 'N/A'), "mAP50": last.get('metrics/mAP50(M)', 'N/A'), "mAP50-95": last.get('metrics/mAP50-95(M)', 'N/A')}
            report += format_table(m_box, m_mask) + "\n"

        # 2. VALIDACIÓN FINAL
        if 'val_metrics' in p:
            report += f"### ✅ Validación Final (Script: `{p.get('script_val', 'N/A')}`)\n"
            report += format_table(p['val_metrics']['box'], p['val_metrics']['mask']) + "\n"

        # 3. TEST / PRUEBAS DE RIGOR
        if 'test_metrics' in p:
            report += f"### 🎯 Test Independiente (Script: `{p.get('script_test', 'N/A')}`)\n"
            report += format_table(p['test_metrics']['box'], p['test_metrics']['mask']) + "\n"
        
        if 'test_rigor_metrics' in p:
            report += f"### 🛡️ Test de Rigor (Script: `{p.get('script_test', 'N/A')}`)\n"
            report += format_table(p['test_rigor_metrics']['box'], p['test_rigor_metrics']['mask']) + "\n"

        # 4. GRÁFICAS Y TRACEABILIDAD
        report += "### 📂 Ubicación de Recursos\n"
        report += f"- **Pesos (Best):** `{p['train_dir']}/weights/best.pt`\n"
        if 'val_dir' in p: report += f"- **Gráficas Validación:** `{p['val_dir']}`\n"
        if 'test_dir' in p: report += f"- **Gráficas Test:** `{p['test_dir']}`\n"
        
        v_train = list_visuals(p['train_dir'])
        important = [v for v in v_train if any(x in v for x in ['F1_curve', 'PR_curve', 'confusion_matrix', 'results.png'])]
        if important:
            report += "- **Gráficas Clave Entrenamiento:** " + ", ".join([f"`{v}`" for v in important]) + "\n"

        report += "\n---\n\n"

    # RESUMEN EJECUTIVO
    report += "## 🏁 RESUMEN COMPARATIVO (mAP50 Mask)\n\n"
    report += "| Experimento | Entrenamiento | Validación Final | Test/Rigor |\n"
    report += "| :--- | :--- | :--- | :--- |\n"
    for p in projects:
        results_csv = os.path.join(p['train_dir'], 'results.csv')
        last = get_last_row_csv(results_csv)
        train_val = last.get('metrics/mAP50(M)', 'N/A') if last else 'N/A'
        v_final = p.get('val_metrics', {}).get('mask', {}).get('mAP50', 'N/A')
        t_final = p.get('test_metrics', {}).get('mask', {}).get('mAP50', 'N/A')
        if t_final == 'N/A': t_final = p.get('test_rigor_metrics', {}).get('mask', {}).get('mAP50', 'N/A')
        
        report += f"| {p['name']} | {train_val} | **{v_final}** | {t_final} |\n"

    with open('CONSOLIDADO_RESULTADOS.md', 'w') as f:
        f.write(report)
    
    print("✅ Dashboard mejorado generado en CONSOLIDADO_RESULTADOS.md")

if __name__ == "__main__":
    main()
