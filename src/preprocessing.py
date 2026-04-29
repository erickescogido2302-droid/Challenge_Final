import pandas as pd
import glob
import os
import time


def preprocess_and_clean(data_path):
    start_time = time.time()
    
    # 1. CARGA RECURSIVA
    search_pattern = os.path.join(data_path, "**", "*.xlsx")
    files = glob.glob(search_pattern, recursive=True)
    all_data = []

    print(f">>> Se detectaron {len(files)} archivos. Iniciando carga...")

    for i, f in enumerate(files, 1):
        try:
            # Mostramos progreso
            print(f"[{i}/{len(files)}] Procesando: {os.path.basename(f)}...", end="\r")
            
            temp_df = pd.read_excel(f)
            temp_df.columns = temp_df.columns.str.upper()
            all_data.append(temp_df)
        except Exception as e:
            print(f"\nError en {f}: {e}")

    print(f"\n>>> Carga completada en {round(time.time() - start_time, 2)} segundos.")
    print(">>> Unificando datos y limpiando...")
    
    df = pd.concat(all_data, ignore_index=True)

    # 2. LIMPIEZA DE DATOS
    # Eliminar columnas con más del 50% de nulos
    limit = len(df) * 0.5
    df = df.dropna(thresh=limit, axis=1)

    # Eliminar columnas constantes (varianza cero)
    df = df.loc[:, df.nunique() > 1]

    # 3. MANEJO DE NULOS (Imputación por moda)
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # 4. EXPORTAR A CSV (Este archivo será tu base para el modelo)
    output_path = os.path.join(data_path, "dataset_limpio.csv")
    print(f">>> Guardando dataset limpio en {output_path}...")
    df.to_csv(output_path, index=False)

    print("\n" + "="*40)
    print("REPORTE DE DATOS LIMPIOS")
    print("="*40)
    print(f"Registros totales: {df.shape[0]}")
    print(f"Columnas finales: {df.shape[1]}")
    print(f"Tiempo total: {round((time.time() - start_time)/60, 2)} minutos.")
    
    return df

if __name__ == "__main__":
    # Asegúrate de que la ruta 'data/' es correcta en tu PC
    df_clean = preprocess_and_clean("data/")