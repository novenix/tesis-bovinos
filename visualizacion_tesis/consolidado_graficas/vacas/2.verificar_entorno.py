#!/usr/bin/env python3
"""
Script de Verificación del Entorno - Pipeline de Curaduría
===========================================================
Verifica que todas las dependencias estén instaladas correctamente
antes de ejecutar el pipeline principal.
"""

import sys
import os
import subprocess
from pathlib import Path


def check_python_version():
    """Verifica versión de Python >= 3.10"""
    version = sys.version_info
    print(f"🐍 Python: {version.major}.{version.minor}.{version.micro}", end=" ")

    if version.major >= 3 and version.minor >= 10:
        print("✅")
        return True
    else:
        print("❌ (Requerido: Python 3.10+)")
        return False


def check_package(package_name, import_name=None):
    """Verifica si un paquete de Python está instalado"""
    import_name = import_name or package_name
    print(f"📦 {package_name}: ", end="")

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'desconocida')
        print(f"✅ (v{version})")
        return True
    except ImportError:
        print(f"❌ NO INSTALADO")
        print(f"   → Instalar con: pip install {package_name}")
        return False


def check_ffmpeg():
    """Verifica instalación de ffmpeg"""
    print("🎬 ffmpeg: ", end="")

    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            # Extraer versión
            first_line = result.stdout.split('\n')[0]
            print(f"✅ ({first_line.split()[2]})")
            return True
        else:
            print("❌ Error ejecutando ffmpeg")
            return False

    except FileNotFoundError:
        print("❌ NO INSTALADO")
        print("   → Instalar con: conda install -c conda-forge ffmpeg")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_directories():
    """Verifica estructura de directorios"""
    print("\n📁 Verificando directorios:")

    dirs = {
        "Datos crudos": Path("/data/estudiantes/vacas/datosCrudos"),
        "Dataset curado": Path("/data/estudiantes/vacas/dataset_curado"),
        "Frames temporales": Path("/data/estudiantes/vacas/frames_temp"),
        "Logs": Path("/data/estudiantes/vacas/logs")
    }

    all_ok = True
    for name, path in dirs.items():
        exists = path.exists()
        writable = path.is_dir() and os.access(path, os.W_OK) if exists else False

        status = "✅" if exists and writable else "⚠️"
        print(f"  {status} {name}: {path}")

        if not exists:
            print(f"     → Se creará automáticamente al ejecutar el pipeline")
        elif not writable:
            print(f"     → ❌ Sin permisos de escritura")
            all_ok = False

    return all_ok


def check_gpu():
    """Verifica disponibilidad de GPU (CUDA)"""
    print("\n🚀 GPU/CUDA:")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✅ GPU disponible: {gpu_name}")
            print(f"  ✅ CUDA version: {torch.version.cuda}")
            print(f"  💡 El pipeline usará GPU automáticamente")
            return True
        else:
            print(f"  ⚠️  GPU no disponible - se usará CPU")
            print(f"  💡 Edita Config.USE_GPU = False en el script para evitar warnings")
            return False
    except ImportError:
        print(f"  ⚠️  PyTorch no instalado - no se puede verificar GPU")
        return False


def test_yolo_world():
    """Prueba de carga del modelo YOLO-World"""
    print("\n🤖 Modelo YOLO-World:")
    print("  Intentando cargar modelo (puede tardar en la primera vez)...")

    try:
        from ultralytics import YOLO

        # Intentar cargar modelo
        model = YOLO("yolov8l-worldv2.pt")
        print("  ✅ Modelo descargado/cargado correctamente")

        # Verificar método set_classes
        if hasattr(model, 'set_classes'):
            model.set_classes(["cow"])
            print("  ✅ Método set_classes() funcional")
        else:
            print("  ⚠️  set_classes() no disponible - actualiza ultralytics")
            return False

        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        print(f"  💡 Si el modelo no existe, se descargará automáticamente")
        return False


def main():
    """Ejecuta todas las verificaciones"""
    print("="*70)
    print("🔍 VERIFICACIÓN DEL ENTORNO - Pipeline de Curaduría")
    print("="*70 + "\n")

    import os

    checks = []

    # Verificaciones críticas
    print("📋 DEPENDENCIAS CRÍTICAS:\n")
    checks.append(("Python 3.10+", check_python_version()))
    checks.append(("ultralytics", check_package("ultralytics")))
    checks.append(("opencv-python", check_package("cv2", "cv2")))
    checks.append(("tqdm", check_package("tqdm")))
    checks.append(("pandas", check_package("pandas")))
    checks.append(("ffmpeg", check_ffmpeg()))

    # Verificaciones adicionales
    checks.append(("Directorios", check_directories()))

    # Verificaciones opcionales
    print("\n📋 VERIFICACIONES OPCIONALES:\n")
    check_gpu()

    # Resumen final
    print("\n" + "="*70)
    critical_ok = all([result for name, result in checks if name != "Directorios"])

    if critical_ok:
        print("✅ ENTORNO LISTO - Puedes ejecutar el pipeline")
        print("\n💡 Comando para ejecutar:")
        print("   python pipeline_curaduria.py")
    else:
        print("❌ ENTORNO INCOMPLETO - Instala las dependencias faltantes")
        print("\n💡 Comando rápido de instalación:")
        print("   conda activate tesis_vacas")
        print("   pip install ultralytics opencv-python-headless tqdm pandas")
        print("   conda install -c conda-forge ffmpeg")

    print("="*70 + "\n")

    return 0 if critical_ok else 1


if __name__ == "__main__":
    sys.exit(main())
