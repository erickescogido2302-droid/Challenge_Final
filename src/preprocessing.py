import pandas as pd
import glob
import os
import time

def ejecutar_limpieza_detective(ruta_data):
    start_time = time.time()
    
    # 1. BUSCAR ARCHIVOS
    search_pattern = os.path.join(ruta_data, "**", "*.xlsx")
    files = glob.glob(search_pattern, recursive=True)
    all_data = []

    print(f"\n>>> Analizando {len(files)} archivos...")

    for f in files:
        try:
            # Leer el Excel
            temp_df = pd.read_excel(f)
            
            # Limpiar nombres de columnas (quitar espacios invisibles)
            temp_df.columns = temp_df.columns.str.strip().str.upper()
            
            # --- EL DETECTIVE ---
            if 'ANO_REG' in temp_df.columns:
                # Ver que años tiene este archivo antes de filtrar
                anios_unicos = temp_df['ANO_REG'].dropna().unique()
                print(f"ARCHIVO: {os.path.basename(f)} | AÑOS DETECTADOS: {list(anios_unicos)[:5]}...")
                all_data.append(temp_df)
            else:
                print(f"!!! ERROR: El archivo {os.path.basename(f)} NO tiene la columna ANO_REG")
                print(f"Columnas encontradas: {list(temp_df.columns)[:5]}")
                
        except Exception as e:
            print(f"No se pudo leer {os.path.basename(f)}: {e}")

    if not all_data:
        print("\n!!! ERROR CRÍTICO: No se cargó ningún dato. Revisa tus archivos Excel.")
        return

    # Unir todo
    df = pd.concat(all_data, ignore_index=True)

    # 2. ASEGURAR QUE LOS AÑOS SEAN NÚMEROS
    df['ANO_REG'] = pd.to_numeric(df['ANO_REG'], errors='coerce')

    # --- AQUÍ ESTÁ LA CORRECCIÓN CLAVE ---
    # Si el año es menor a 100 (ej. 85, 90, 97), le sumamos 1900
    df.loc[df['ANO_REG'] < 100, 'ANO_REG'] += 1900
    print(">>> Se normalizaron años de 2 dígitos (ej. 85 -> 1985)")
    # -------------------------------------

    # 3. FILTRAR (Ahora sí incluirá desde 1985 correctamente)
    df_filtrado = df[(df['ANO_REG'] >= 1985) & (df['ANO_REG'] <= 2024)].copy()

    # 4. GUARDAR
    output_path = os.path.join(ruta_data, "dataset_limpio.csv")
    df_filtrado.to_csv(output_path, index=False)

    print("\n" + "="*50)
    print("                REPORTE FINAL")
    print("="*50)
    print(f"Registros encontrados en total: {len(df):,}")
    print(f"Registros que sobrevivieron al filtro (1985-2024): {len(df_filtrado):,}")
    
    # Verificamos si realmente hay datos viejos
    anios_finales = sorted(df_filtrado['ANO_REG'].unique().astype(int))
    print(f"Años que quedaron en el archivo final: {anios_finales}")
    print("="*50)

# --- EJECUCIÓN DIRECTA ---
if __name__ == "__main__":
    # Detectar carpeta
    ruta = "data/" if os.path.exists("data/") else "../data/"
    print(f"Iniciando limpieza en: {os.path.abspath(ruta)}")
    ejecutar_limpieza_detective(ruta)