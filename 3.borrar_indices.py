import os

# --- CONFIGURACIÓN ---
# Ruta exacta donde están tus datos crudos
ruta_base = "/data/estudiantes/vacas/datosCrudos"
# Extensión a borrar
extension_a_borrar = ".idx"

print(f"--- INICIANDO LIMPIEZA EN: {ruta_base} ---")

archivos_encontrados = 0
archivos_borrados = 0

# 1. Primero contamos qué hay (Modo Seguro)
print("Buscando archivos...")
archivos_para_borrar = []

for root, dirs, files in os.walk(ruta_base):
    for file in files:
        if file.endswith(extension_a_borrar):
            ruta_completa = os.path.join(root, file)
            archivos_para_borrar.append(ruta_completa)

print(f"Se encontraron {len(archivos_para_borrar)} archivos con extensión {extension_a_borrar}.")

# 2. Confirmación (Opcional, pero recomendado en scripts manuales)
if len(archivos_para_borrar) > 0:
    confirmacion = input("¿Deseas proceder a BORRARLOS permanentemente? (escribe 'si' para confirmar): ")
    
    if confirmacion.lower() == 'si':
        for ruta in archivos_para_borrar:
            try:
                os.remove(ruta)
                # print(f"Borrado: {ruta}") # Descomenta para ver lista detallada
                archivos_borrados += 1
            except OSError as e:
                print(f"Error borrando {ruta}: {e}")
        print(f"--- LISTO: Se borraron {archivos_borrados} archivos. ---")
    else:
        print("Operación cancelada. No se borró nada.")
else:
    print("No hay archivos .idx para borrar.")