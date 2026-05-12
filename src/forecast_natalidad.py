import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# 1. Cargar y procesar datos
try:
    df = pd.read_csv('data/dataset_limpio.csv')
    col_anio = 'ANO_REG'
    
    # Agrupamos y filtramos años mentirosos
    conteo_anual = df.groupby(col_anio).size().reset_index(name='Nacimientos')
    conteo_anual = conteo_anual[conteo_anual['Nacimientos'] > 20000].copy()

    # --- BLOQUE DE INTERPOLACIÓN (PARA LOS HUECOS) ---
    # Creamos un rango completo de años desde 1985 hasta 2024
    rango_completo = pd.DataFrame({col_anio: np.arange(1985, 2025)})
    # Unimos con nuestros datos reales
    df_completo = pd.merge(rango_completo, conteo_anual, on=col_anio, how='left')
    # Interpolamos los valores faltantes basándonos en la línea
    df_completo['Nacimientos_Estimados'] = df_completo['Nacimientos'].interpolate(method='linear')
    # -------------------------------------------------
    
    conteo_anual = df_completo.copy()

except Exception as e:
    print(f"Error: {e}")
    exit()

# 2. Modelo de Predicción (usando los datos reales más recientes)
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

# Graficar la línea completa (usando estimaciones para los huecos)
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
    
    # Si es un dato real, texto normal. Si es estimado (era NaN), texto en cursiva/gris
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
plt.title('Evolución de Natalidad en Guadalajara con Estimación de Vacíos (1985 - 2026)', fontsize=18, fontweight='bold')
plt.xlabel('Año de Registro', fontsize=12)
plt.ylabel('Cantidad de Nacimientos', fontsize=12)
plt.xticks(np.arange(1985, 2027, 1), rotation=45)
plt.grid(True, linestyle=':', alpha=0.5)
plt.xlim(1984, 2027)
plt.ylim(0, max(todos_y if 'todos_y' in locals() else [60000]) * 1.2)
plt.legend(loc='upper right')
plt.tight_layout()

plt.savefig('results/natalidad_guadalajara_final_estimada.png', dpi=300)
plt.show()