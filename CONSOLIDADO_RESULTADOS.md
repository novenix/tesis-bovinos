# 📊 DASHBOARD INTEGRAL DE MÉTRICAS - PROYECTO DE GRADO

Este reporte consolida todos los resultados de entrenamiento, validación final y pruebas de rigor para Bovinos y Moscas.

## 🚀 Bovinos v16 (Nano)
- **Objetivo:** Bovinos Segmentation
- **Directorio:** `runs/segment/tesis_bovinos_yolo26/train2`
### ⚙️ Configuración del Modelo
| Modelo Base | Tamaño Imagen | Batch | Epochs | Dataset |
| :--- | :--- | :--- | :--- | :--- |
| `yolo26n-seg.pt` | `640` | `80` | `100` | `dataset_v2.yaml` |

### 📈 Resultados de Entrenamiento (Último Epoch)
| Tipo | Precisión | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| **Box** | 0.55292 | 0.33453 | 0.37009 | 0.22909 |
| **Mask** | 0.40296 | 0.19685 | 0.19411 | 0.10502 |

### ✅ Validación Final (Script: `16_17.1.validar_modelos.py`)
| Tipo | Precisión | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| **Box** | 0.554 | 0.334 | 0.370 | 0.229 |
| **Mask** | 0.414 | 0.186 | 0.186 | 0.095 |

### 🛡️ Test de Rigor (Script: `16.4.preparar_test_rigor_nano.py`)
| Tipo | Precisión | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| **Box** | 0.165 | 0.098 | 0.084 | 0.050 |
| **Mask** | 0.199 | 0.056 | 0.046 | 0.024 |

### 📂 Ubicación de Recursos
- **Pesos (Best):** `runs/segment/tesis_bovinos_yolo26/train2/weights/best.pt`
- **Gráficas Validación:** `runs/segment/runs/segment/tesis_bovinos_yolo26/train2/validacion_final_v16`
- **Gráficas Clave Entrenamiento:** `BoxF1_curve.png`, `BoxPR_curve.png`, `MaskF1_curve.png`, `MaskPR_curve.png`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `results.png`

---

## 🚀 Bovinos v17 (Medium)
- **Objetivo:** Bovinos Segmentation
- **Directorio:** `runs/segment/tesis_bovinos_medium/train_v17_medium`
### ⚙️ Configuración del Modelo
| Modelo Base | Tamaño Imagen | Batch | Epochs | Dataset |
| :--- | :--- | :--- | :--- | :--- |
| `yolo26m-seg.pt` | `640` | `40` | `100` | `dataset_v3.yaml` |

### 📈 Resultados de Entrenamiento (Último Epoch)
| Tipo | Precisión | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| **Box** | 0.67263 | 0.42954 | 0.48874 | 0.34342 |
| **Mask** | 0.45817 | 0.26747 | 0.25943 | 0.14453 |

### ✅ Validación Final (Script: `16_17.1.validar_modelos.py`)
| Tipo | Precisión | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| **Box** | 0.672 | 0.429 | 0.489 | 0.344 |
| **Mask** | 0.461 | 0.253 | 0.246 | 0.125 |

### 🛡️ Test de Rigor (Script: `17.2.preparar_test_rigor_medium.py`)
| Tipo | Precisión | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| **Box** | 0.290 | 0.168 | 0.172 | 0.114 |
| **Mask** | 0.179 | 0.101 | 0.087 | 0.040 |

### 📂 Ubicación de Recursos
- **Pesos (Best):** `runs/segment/tesis_bovinos_medium/train_v17_medium/weights/best.pt`
- **Gráficas Validación:** `runs/segment/runs/segment/tesis_bovinos_medium/train_v17_medium/validacion_final_v17`
- **Gráficas Clave Entrenamiento:** `BoxF1_curve.png`, `BoxPR_curve.png`, `MaskF1_curve.png`, `MaskPR_curve.png`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `results.png`

---

## 🚀 Moscas v1 (Medium)
- **Objetivo:** Moscas Segmentation
- **Directorio:** `runs/segment/entrenamiento_manual_v27/yolo26_moscas_v1_medium`
### ⚙️ Configuración del Modelo
| Modelo Base | Tamaño Imagen | Batch | Epochs | Dataset |
| :--- | :--- | :--- | :--- | :--- |
| `yolo26m-seg.pt` | `640` | `48` | `100` | `data.yaml` |

### 📈 Resultados de Entrenamiento (Último Epoch)
| Tipo | Precisión | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| **Box** | 0.34566 | 0.34624 | 0.25595 | 0.07263 |
| **Mask** | 0.28926 | 0.29233 | 0.19796 | 0.05425 |

### ✅ Validación Final (Script: `N/A`)
| Tipo | Precisión | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| **Box** | 0.341 | 0.335 | 0.251 | 0.069 |
| **Mask** | 0.336 | 0.322 | 0.242 | 0.068 |

### 📂 Ubicación de Recursos
- **Pesos (Best):** `runs/segment/entrenamiento_manual_v27/yolo26_moscas_v1_medium/weights/best.pt`
- **Gráficas Clave Entrenamiento:** `BoxF1_curve.png`, `BoxPR_curve.png`, `MaskF1_curve.png`, `MaskPR_curve.png`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `results.png`

---

## 🚀 Moscas v2 (Pequeño)
- **Objetivo:** Moscas Segmentation
- **Directorio:** `runs/segment/entrenamiento_manual_v27/yolo26_moscas_v2_pequeño_1024`
### ⚙️ Configuración del Modelo
| Modelo Base | Tamaño Imagen | Batch | Epochs | Dataset |
| :--- | :--- | :--- | :--- | :--- |
| `yolo26m-seg.pt` | `1024` | `12` | `150` | `data.yaml` |

### 📈 Resultados de Entrenamiento (Último Epoch)
| Tipo | Precisión | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| **Box** | 0.64245 | 0.63795 | 0.62063 | 0.21196 |
| **Mask** | 0.5272 | 0.54469 | 0.47768 | 0.14091 |

### ✅ Validación Final (Script: `N/A`)
| Tipo | Precisión | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| **Box** | 0.537 | 0.467 | 0.421 | 0.136 |
| **Mask** | 0.489 | 0.489 | 0.415 | 0.109 |

### 🎯 Test Independiente (Script: `27.4.test_moscas_v2`)
| Tipo | Precisión | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| **Box** | 0.425 | 0.487 | 0.362 | 0.102 |
| **Mask** | 0.384 | 0.457 | 0.296 | 0.069 |

### 📂 Ubicación de Recursos
- **Pesos (Best):** `runs/segment/entrenamiento_manual_v27/yolo26_moscas_v2_pequeño_1024/weights/best.pt`
- **Gráficas Validación:** `runs/segment/validacion_manual_v27/val_moscas_v2_pequeño_1024`
- **Gráficas Test:** `runs/segment/test_manual_v27/test_moscas_v2_pequeño_1024`
- **Gráficas Clave Entrenamiento:** `BoxF1_curve.png`, `BoxPR_curve.png`, `MaskF1_curve.png`, `MaskPR_curve.png`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `results.png`

---

## 🚀 Moscas v3 (Micro)
- **Objetivo:** Moscas Segmentation
- **Directorio:** `runs/segment/entrenamiento_manual_v27/yolo26_moscas_v3_micro_1024`
### ⚙️ Configuración del Modelo
| Modelo Base | Tamaño Imagen | Batch | Epochs | Dataset |
| :--- | :--- | :--- | :--- | :--- |
| `yolo26m-seg.pt` | `1024` | `12` | `150` | `data.yaml` |

### 📈 Resultados de Entrenamiento (Último Epoch)
| Tipo | Precisión | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| **Box** | 0.57146 | 0.58873 | 0.53128 | 0.16881 |
| **Mask** | 0.46836 | 0.51425 | 0.41297 | 0.11867 |

### ✅ Validación Final (Script: `N/A`)
| Tipo | Precisión | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| **Box** | 0.565 | 0.577 | 0.530 | 0.171 |
| **Mask** | 0.478 | 0.516 | 0.426 | 0.122 |

### 📂 Ubicación de Recursos
- **Pesos (Best):** `runs/segment/entrenamiento_manual_v27/yolo26_moscas_v3_micro_1024/weights/best.pt`
- **Gráficas Clave Entrenamiento:** `BoxF1_curve.png`, `BoxPR_curve.png`, `MaskF1_curve.png`, `MaskPR_curve.png`, `confusion_matrix.png`, `confusion_matrix_normalized.png`, `results.png`

---

## 🏁 RESUMEN COMPARATIVO (mAP50 Mask)

| Experimento | Entrenamiento | Validación Final | Test/Rigor |
| :--- | :--- | :--- | :--- |
| Bovinos v16 (Nano) | 0.19411 | **0.186** | 0.046 |
| Bovinos v17 (Medium) | 0.25943 | **0.246** | 0.087 |
| Moscas v1 (Medium) | 0.19796 | **0.242** | N/A |
| Moscas v2 (Pequeño) | 0.47768 | **0.415** | 0.296 |
| Moscas v3 (Micro) | 0.41297 | **0.426** | N/A |

---

## 🎨 VISUALIZACIONES ACADÉMICAS (Vectoriales)

Se han generado gráficas en formato de alta calidad (SVG/PDF) listas para ser incluidas en el documento de tesis. Se encuentran en: `visualizacion_tesis/consolidado_graficas/`

### 1. Comparativas Globales de Desempeño
*   **mAP50 (Mask):** `comparativa_modelos_map50.pdf / .svg`
*   **Precisión (Mask):** `comparativa_modelos_precision.pdf / .svg`
*   **Recall (Mask):** `comparativa_modelos_recall.pdf / .svg`
*   **Descripción:** Gráficas de barras que comparan las métricas clave de todos los experimentos en el set de validación final. Ideal para justificar el balance entre detección y segmentación.

### 2. Análisis de Generalización (Degradación)
*   **Archivo:** `degradacion_val_vs_test.pdf / .svg`
*   **Descripción:** Comparativa directa entre el set de Validación y el set de Test/Rigor. Permite visualizar qué tanto "cae" el modelo al enfrentarse a datos totalmente nuevos (rigor).

### 3. Balance de Métricas (Radar Charts)
*   **Archivos:** `radar_[modelo].pdf / .svg`
*   **Descripción:** Gráficas de araña que muestran el balance entre **Precisión**, **Recall** y **mAP50**.
    *   `radar_bovinos_v16_nano`
    *   `radar_bovinos_v17_medium`
    *   `radar_moscas_v1_medium`
    *   `radar_moscas_v2_pequeño`
    *   `radar_moscas_v3_micro`
