#!/bin/bash
################################################################################
# Script de Instalación Automática - Pipeline de Curaduría
################################################################################
# Este script configura todo el entorno necesario para ejecutar el pipeline
# de detección de ganado sin permisos sudo.
#
# Uso:
#   bash setup_entorno.sh
#
# O con permisos de ejecución:
#   chmod +x setup_entorno.sh
#   ./setup_entorno.sh
################################################################################

set -e  # Detener si hay error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funciones de utilidad
print_header() {
    echo ""
    echo "================================================================================"
    echo -e "${BLUE}$1${NC}"
    echo "================================================================================"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Verificar que conda esté disponible
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda no encontrado en PATH"
        echo ""
        echo "Por favor, instala Anaconda/Miniconda primero:"
        echo "  https://docs.conda.io/en/latest/miniconda.html"
        echo ""
        echo "O si ya está instalado, actívalo:"
        echo "  source ~/anaconda3/bin/activate"
        echo "  source ~/miniconda3/bin/activate"
        exit 1
    fi
    print_success "Conda encontrado: $(conda --version)"
}

# Banner inicial
clear
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                🐄 INSTALACIÓN AUTOMÁTICA - PIPELINE DE CURADURÍA 🐄           ║
║                                                                               ║
║                     Sistema de Detección de Ganado con YOLO-World            ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
EOF

echo ""
print_info "Este script instalará:"
echo "  • Entorno Anaconda (tesis_vacas)"
echo "  • ffmpeg (procesamiento de video)"
echo "  • ultralytics (YOLO-World)"
echo "  • opencv, tqdm, pandas (dependencias)"
echo ""
echo "Ubicación: /data/estudiantes/vacas"
echo "Tiempo estimado: 5-10 minutos"
echo ""

# Confirmar antes de continuar
read -p "¿Continuar con la instalación? (s/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[SsYy]$ ]]; then
    print_warning "Instalación cancelada por el usuario"
    exit 0
fi

# PASO 1: Verificar conda
print_header "PASO 1/6: Verificando Conda"
check_conda

# PASO 2: Crear entorno
print_header "PASO 2/6: Creando entorno tesis_vacas"

if conda env list | grep -q "tesis_vacas"; then
    print_warning "El entorno 'tesis_vacas' ya existe"
    read -p "¿Eliminarlo y recrear? (s/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[SsYy]$ ]]; then
        print_info "Eliminando entorno existente..."
        conda env remove -n tesis_vacas -y
        print_success "Entorno eliminado"
    else
        print_info "Usando entorno existente"
    fi
fi

if ! conda env list | grep -q "tesis_vacas"; then
    print_info "Creando nuevo entorno con Python 3.10..."
    conda create -n tesis_vacas python=3.10 -y
    print_success "Entorno creado"
else
    print_success "Entorno verificado"
fi

# Activar entorno
print_info "Activando entorno tesis_vacas..."
eval "$(conda shell.bash hook)"
conda activate tesis_vacas
print_success "Entorno activado"

# PASO 3: Instalar ffmpeg
print_header "PASO 3/6: Instalando ffmpeg"

if command -v ffmpeg &> /dev/null; then
    print_warning "ffmpeg ya está instalado: $(ffmpeg -version | head -n 1)"
    read -p "¿Reinstalar de todas formas? (s/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[SsYy]$ ]]; then
        print_info "Saltando instalación de ffmpeg"
    else
        print_info "Reinstalando ffmpeg desde conda-forge..."
        conda install -c conda-forge ffmpeg --force-reinstall -y
        print_success "ffmpeg reinstalado"
    fi
else
    print_info "Instalando ffmpeg desde conda-forge..."
    conda install -c conda-forge ffmpeg -y
    print_success "ffmpeg instalado"
fi

# Verificar ffmpeg
if command -v ffmpeg &> /dev/null; then
    print_success "ffmpeg verificado: $(ffmpeg -version | head -n 1 | cut -d' ' -f3)"
else
    print_error "ffmpeg no se instaló correctamente"
    exit 1
fi

# PASO 4: Instalar paquetes de Python
print_header "PASO 4/6: Instalando paquetes de Python"

print_info "Instalando ultralytics (YOLO-World)..."
pip install ultralytics==8.1.24 --quiet
print_success "ultralytics instalado"

print_info "Instalando opencv-python-headless..."
pip install opencv-python-headless --quiet
print_success "opencv-python-headless instalado"

print_info "Instalando tqdm, pandas, Pillow..."
pip install tqdm pandas Pillow --quiet
print_success "Dependencias instaladas"

# PASO 5: Verificar instalación
print_header "PASO 5/6: Verificando instalación"

print_info "Ejecutando script de verificación..."
echo ""

cd /data/estudiantes/vacas
python verificar_entorno.py

# PASO 6: Crear directorios
print_header "PASO 6/6: Creando estructura de directorios"

DIRS=(
    "/data/estudiantes/vacas/dataset_curado"
    "/data/estudiantes/vacas/frames_temp"
    "/data/estudiantes/vacas/logs"
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_success "Creado: $dir"
    else
        print_info "Ya existe: $dir"
    fi
done

# Verificar directorio de datos crudos
if [ ! -d "/data/estudiantes/vacas/datosCrudos" ]; then
    print_warning "Directorio datosCrudos NO existe"
    print_info "Creando directorio vacío (debes copiar tus videos aquí)..."
    mkdir -p /data/estudiantes/vacas/datosCrudos
    print_success "Creado: /data/estudiantes/vacas/datosCrudos"
    echo ""
    print_warning "¡IMPORTANTE! Copia tus videos .dav a:"
    echo "  /data/estudiantes/vacas/datosCrudos/"
fi

# Dar permisos de ejecución a scripts
chmod +x /data/estudiantes/vacas/pipeline_curaduria.py
chmod +x /data/estudiantes/vacas/test_single_video.py
chmod +x /data/estudiantes/vacas/verificar_entorno.py
print_success "Permisos de ejecución configurados"

# Resumen final
print_header "✅ INSTALACIÓN COMPLETADA"

cat << EOF
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          INSTALACIÓN EXITOSA                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📁 Estructura de archivos:
   /data/estudiantes/vacas/
   ├── pipeline_curaduria.py          [Script principal]
   ├── test_single_video.py           [Test con un video]
   ├── verificar_entorno.py           [Verificación de dependencias]
   ├── setup_entorno.sh               [Este script]
   ├── README.md                      [Documentación completa]
   ├── datosCrudos/                   [Videos .dav (INPUT)]
   ├── dataset_curado/                [Frames con ganado (OUTPUT)]
   ├── frames_temp/                   [Temporal]
   └── logs/                          [Reportes JSON/CSV]

🚀 Próximos pasos:

   1. Activar el entorno (en cada sesión):
      ${GREEN}conda activate tesis_vacas${NC}

   2. Copiar videos .dav (si no lo hiciste aún):
      ${GREEN}cp /ruta/origen/*.dav /data/estudiantes/vacas/datosCrudos/${NC}

   3. Test con un solo video (recomendado primero):
      ${GREEN}python test_single_video.py datosCrudos/tu_video.dav${NC}

   4. Ejecutar pipeline completo:
      ${GREEN}python pipeline_curaduria.py${NC}

   5. Ver resultados:
      ${GREEN}ls -lh dataset_curado/
      cat logs/*.csv${NC}

📚 Documentación:
   ${GREEN}cat README.md${NC}

💡 Ajustar parámetros:
   Edita la clase Config en pipeline_curaduria.py (línea 30)
   - FRAME_INTERVAL_SECONDS (default: 3)
   - CONFIDENCE_THRESHOLD (default: 0.4)
   - USE_GPU (default: True)

🐛 Problemas:
   ${GREEN}python verificar_entorno.py${NC}

EOF

# Información sobre GPU
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    print_success "GPU CUDA detectada - El pipeline usará aceleración por hardware"
else
    print_warning "GPU no detectada - El pipeline usará CPU (será más lento)"
    print_info "Para usar GPU, instala PyTorch con CUDA:"
    echo "  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
fi

echo ""
print_success "¡Todo listo para comenzar! 🎉"
echo ""

# Instrucción final
print_info "Para activar el entorno en el futuro, usa:"
echo -e "  ${GREEN}conda activate tesis_vacas${NC}"
echo ""
