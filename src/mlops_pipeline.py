import subprocess
import sys

def run_script(script_name):
    """Ejecuta un script de python y verifica si terminó correctamente."""
    print(f"\n>>> Ejecutando: {script_name}...")
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    if result.returncode == 0:
        print(f"--- {script_name} completado con éxito. ---")
    else:
        print(f"!!! Error en {script_name}. Revisa los logs. !!!")
        sys.exit(1)

if __name__ == "__main__":
    print("==========================================")
    print("INICIANDO PIPELINE DE MLOPS - NATALIDAD")
    print("==========================================\n")

    # 1. Limpieza y preparación de datos
    run_script("src/preprocessing.py")

    # 2. Entrenamiento y Registro en MLflow
    run_script("src/model_training.py")

    print("\n==========================================")
    print("PIPELINE FINALIZADO EXITOSAMENTE")
    print("==========================================")