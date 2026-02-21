import cv2
import os
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
import time
import sys

def rotate_image(img_path):
    try:
        # Leer la imagen
        img = cv2.imread(str(img_path))
        if img is None:
            return f"Error: No se pudo leer {img_path}"
        
        # Rotar 90 grados a la derecha (sentido horario)
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        
        # Guardar la imagen (sobreescribiendo la original como se pidió)
        cv2.imwrite(str(img_path), rotated_img)
        return None
    except Exception as e:
        return f"Error procesando {img_path}: {str(e)}"

def main():
    test_mode = "--test" in sys.argv
    image_dir = Path("dataset_curado/train/images")
    if not image_dir.exists():
        print(f"Error: El directorio {image_dir} no existe.")
        return

    # Obtener lista de imágenes
    print(f"Buscando imágenes en {image_dir}...")
    image_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]
    
    if test_mode:
        image_files = image_files[:100]
        print("MODO TEST: Procesando solo 100 imágenes.")
    
    total_images = len(image_files)
    print(f"Total de imágenes a procesar: {total_images}")

    if total_images == 0:
        print("No se encontraron imágenes para rotar.")
        return

    # Usar ThreadPoolExecutor
    max_workers = 24 # Ajustado para balancear CPU e I/O
    print(f"Iniciando rotación con {max_workers} hilos...")

    errors = []
    start_time = time.time()
    
    with tqdm(total=total_images, desc="Rotando imágenes") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_img = {executor.submit(rotate_image, img_path): img_path for img_path in image_files}
            for future in concurrent.futures.as_completed(future_to_img):
                result = future.result()
                if result:
                    errors.append(result)
                pbar.update(1)

    end_time = time.time()
    duration = end_time - start_time
    
    if test_mode:
        avg_time = duration / total_images
        estimated_total = avg_time * 173956 / 60 # en minutos
        print(f"\nPrueba completada en {duration:.2f}s (Promedio: {avg_time:.4f}s por imagen)")
        print(f"Tiempo total estimado para 173,956 imágenes: {estimated_total:.2f} minutos ({estimated_total/60:.2f} horas)")
    else:
        print(f"\nProceso finalizado en {duration/60:.2f} minutos.")

    if errors:
        print(f"Se encontraron {len(errors)} errores.")
        with open("logs/errores_rotacion.log", "a") as f:
            for error in errors:
                f.write(f"{error}\n")
    else:
        if not test_mode:
            print("¡Rotación completada exitosamente!")

if __name__ == "__main__":
    main()
