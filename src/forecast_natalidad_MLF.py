import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import mlflow
import os

# --- CONFIGURACIÓN MLFLOW ---
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Proyeccion_Natalidad_GDL")

# 1. Cargar y procesar datos
try:
    df = pd.read_csv('data/dataset_limpio.csv')
    col_anio = 'ANO_REG'
    
    # Agrupamos y filtramos años
    conteo_anual = df.groupby(col_anio).size().reset_index(name='Nacimientos')
    conteo_anual = conteo_anual[conteo_anual['Nacimientos'] > 20000].copy()

    # --- BLOQUE DE INTERPOLACIÓN ---
    rango_completo = pd.DataFrame({col_anio: np.arange(1985, 2025)})
    df_completo = pd.merge(rango_completo, conteo_anual, on=col_anio, how='left')
    df_completo['Nacimientos_Estimados'] = df_completo['Nacimientos'].interpolate(method='linear')
    conteo_anual = df_completo.copy()

except Exception as e:
    print(f"Error: {e}")
    exit()

# Iniciar el registro en MLflow
with mlflow.start_run(run_name="Regresion_Lineal_Proyeccion"):
    
    # 2. Modelo de Predicción (AHORA INDENTADO)
    datos_reales = conteo_anual.dropna(subset=['Nacimientos'])
    datos_recientes = datos_reales[datos_reales[col_anio] >= 2014]
    X_train = datos_recientes[[col_anio]].values
    y_train = datos_recientes['Nacimientos'].values

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    anios_futuros = np.array([[2025], [2026]])
    predicciones = modelo.predict(anios_futuros)

    # 3. Gráfica
    plt.figure(figsize=(20, 10))

    # Graficar la línea completa
    plt.plot(conteo_anual[col_anio], conteo_anual['Nacimientos_Estimados'], 
             marker='o', color='#34495e', label='Tendencia Histórica (Estimada en huecos)', linewidth=2.5)

    # Graficar Proyección
    eje_x_proy = [2024, 2025, 2026]
    eje_y_proy = [conteo_anual['Nacimientos_Estimados'].iloc[-1], predicciones[0], predicciones[1]]
    plt.plot(eje_x_proy, eje_y_proy, marker='s', linestyle='--', color='#e74c3c', label='Proyección GDL', linewidth=2.5)

    # --- ETIQUETADO INTELIGENTE ---
    for i, row in conteo_anual.iterrows():
        anio = int(row[col_anio])
        valor = row['Nacimientos_Estimados']
        
        if pd.isna(row['Nacimientos']):
            plt.text(anio, valor + 800, f'~{int(valor):,}', 
                     ha='center', va='bottom', fontsize=8, color='#7f8c8d', rotation=90, style='italic')
        else:
            plt.text(anio, valor + 800, f'{int(valor):,}', 
                     ha='center', va='bottom', fontsize=9, color='#34495e', fontweight='bold', rotation=90)

    # Etiquetas de proyección
    for x, y in zip([2025, 2026], predicciones):
        plt.text(x, y + 800, f'{int(y):,}', ha='center', va='bottom', fontsize=9, color='#e74c3c', fontweight='bold', rotation=90)

    # 4. Formato final
    plt.title('Evolución de Natalidad en Guadalajara (1985 - 2026)', fontsize=18, fontweight='bold')
    plt.xlabel('Año de Registro', fontsize=12)
    plt.ylabel('Cantidad de Nacimientos', fontsize=12)
    plt.xticks(np.arange(1985, 2027, 1), rotation=45)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.xlim(1984, 2027)
    
    # Corregido: cálculo del límite superior de Y
    max_nacimientos = max(conteo_anual['Nacimientos_Estimados'].max(), predicciones.max())
    plt.ylim(0, max_nacimientos * 1.3)
    
    plt.legend(loc='upper right')
    plt.tight_layout()

    # Guardar localmente
    os.makedirs('results', exist_ok=True)
    ruta_grafica = 'results/natalidad_guadalajara_final_estimada.png'
    plt.savefig(ruta_grafica, dpi=300)

    # --- REGISTRO EN MLFLOW ---
    mlflow.log_artifact(ruta_grafica) # Sube la imagen a Artefactos
    mlflow.log_metric("prediccion_2025", predicciones[0])
    mlflow.log_metric("prediccion_2026", predicciones[1])
    mlflow.log_param("modelo", "LinearRegression")
    
    print("\n>>> Entrenamiento exitoso. Datos y gráfica enviados a MLflow.")
    
    # plt.show() # Opcional: puedes dejarlo o comentarlo si solo quieres verlo en MLflow
    plt.close()