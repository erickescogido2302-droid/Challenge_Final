import pandas as pd
import os

def ver_listado_por_anio():
    # Detectar ruta
    ruta = "data/dataset_limpio.csv" if os.path.exists("data/") else "../data/dataset_limpio.csv"

    if not os.path.exists(ruta):
        print(f"!!! Error: No se encontró el archivo en {ruta}")
        return

    # Leer el dataset (usamos sep=',' porque el nuevo preprocessing lo guardó así)
    df = pd.read_csv(ruta)

    print("\n" + "="*40)
    print("   CONTEO DETALLADO POR AÑO (REGISTROS)")
    print("="*40)

    if 'ANO_REG' in df.columns:
        # Contamos cuántas veces aparece cada año y ordenamos
        conteo = df['ANO_REG'].value_counts().sort_index()
        
        print(f"{'AÑO':<10} | {'TOTAL REGISTROS':<15}")
        print("-" * 30)
        
        for anio, total in conteo.items():
            # El int(anio) es para quitar el .0 si es que aparece
            print(f"{int(anio):<10} | {total:>15,}")
            
        print("-" * 30)
        print(f"{'TOTAL':<10} | {conteo.sum():>15,}")
    else:
        print("Error: No se encontró la columna 'ANO_REG'.")
        print(f"Columnas disponibles: {df.columns.tolist()}")

    print("="*40)

if __name__ == "__main__":
    ver_listado_por_anio()