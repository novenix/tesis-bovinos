import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from math import pi

# Configuración de estilo académico
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
plt.rcParams.update({
    'font.family': 'serif',
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
})

output_dir = "visualizacion_tesis/consolidado_graficas"
os.makedirs(output_dir, exist_ok=True)

# Datos Consolidados
data = [
    {"Model": "Bovinos v16 (Nano)", "Task": "Bovinos", "Val_mAP50_Mask": 0.186, "Test_mAP50_Mask": 0.046, "P_Mask": 0.414, "R_Mask": 0.186},
    {"Model": "Bovinos v17 (Medium)", "Task": "Bovinos", "Val_mAP50_Mask": 0.246, "Test_mAP50_Mask": 0.087, "P_Mask": 0.461, "R_Mask": 0.253},
    {"Model": "Moscas v1 (Medium)", "Task": "Moscas", "Val_mAP50_Mask": 0.242, "Test_mAP50_Mask": None, "P_Mask": 0.336, "R_Mask": 0.322},
    {"Model": "Moscas v2 (Pequeño)", "Task": "Moscas", "Val_mAP50_Mask": 0.415, "Test_mAP50_Mask": 0.296, "P_Mask": 0.489, "R_Mask": 0.489},
    {"Model": "Moscas v3 (Micro)", "Task": "Moscas", "Val_mAP50_Mask": 0.426, "Test_mAP50_Mask": None, "P_Mask": 0.478, "R_Mask": 0.516},
]

df = pd.DataFrame(data)

# --- GRÁFICA 1: COMPARATIVA mAP50 MASK (TODOS LOS MODELOS) ---
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="Val_mAP50_Mask", data=df, palette="viridis")
plt.title("Comparativa de Desempeño mAP50 (Mask) - Validación")
plt.ylabel("mAP50 (Mask)")
plt.xlabel("Modelo")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"{output_dir}/comparativa_modelos_map50.pdf")
plt.savefig(f"{output_dir}/comparativa_modelos_map50.svg")
plt.close()

# --- GRÁFICA 2: DEGRADACIÓN VAL vs TEST (MODELOS DISPONIBLES) ---
df_deg = df.dropna(subset=['Test_mAP50_Mask']).melt(id_vars=["Model"], value_vars=["Val_mAP50_Mask", "Test_mAP50_Mask"], var_name="Set", value_name="mAP50")
df_deg['Set'] = df_deg['Set'].replace({"Val_mAP50_Mask": "Validación", "Test_mAP50_Mask": "Test/Rigor"})

plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="mAP50", hue="Set", data=df_deg, palette="muted")
plt.title("Análisis de Generalización: Validación vs Test/Rigor")
plt.ylabel("mAP50 (Mask)")
plt.xlabel("Modelo")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"{output_dir}/degradacion_val_vs_test.pdf")
plt.savefig(f"{output_dir}/degradacion_val_vs_test.svg")
plt.close()

# --- GRÁFICA 3: RADAR CHART (BALANCE P, R, mAP) ---
def make_radar_chart(row, title, filename):
    categories = ['Precisión', 'Recall', 'mAP50']
    values = [row['P_Mask'], row['R_Mask'], row['Val_mAP50_Mask']]
    
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]
    
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)
    plt.title(title, size=15, y=1.1)
    plt.savefig(filename)
    plt.close()

# Generar un radar individual por cada modelo para claridad académica
for _, row in df.iterrows():
    name_clean = row['Model'].replace(" ", "_").replace("(", "").replace(")", "").lower()
    make_radar_chart(row, f"Balance de Métricas: {row['Model']}", f"{output_dir}/radar_{name_clean}.pdf")
    make_radar_chart(row, f"Balance de Métricas: {row['Model']}", f"{output_dir}/radar_{name_clean}.svg")

print(f"✅ Gráficas académicas generadas en {output_dir}")
