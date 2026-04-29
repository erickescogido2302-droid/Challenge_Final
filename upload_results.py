import os

def upload_changes():
    # 1. Asegurar que Git LFS esté activo para el archivo pesado
    print("Configurando Git LFS...")
    os.system('git lfs track "data/dataset_limpio.csv"')
    os.system('git add .gitattributes')

    # 2. Añadir todos los cambios (scripts, imágenes y el CSV)
    print("Añadiendo archivos...")
    os.system('git add --all')

    # 3. Hacer el commit
    # Usamos comillas dobles para el mensaje por si hay espacios
    print("Creando commit...")
    os.system('git commit -m "Actualización de resultados y dataset vía script"')

    # 4. Subir a GitHub
    print("Subiendo a GitHub (esto puede tardar por el tamaño del archivo)...")
    result = os.system('git push origin main')

    if result == 0:
        print("\n¡Éxito! Todo se subió correctamente.")
    else:
        print("\nError al subir. Revisa si hay conflictos en el repositorio.")

if __name__ == "__main__":
    upload_changes()