import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# 1. Cargar y filtrar para Guadalajara (Municipio)
try:
    df = pd.read_csv('data/dataset_limpio.zip', compression='zip')
    col_anio = 'ANO_NAC'
    
    # Filtro: 1985-2024 y aseguramos que los datos sean coherentes
    df = df[(df[col_anio] >= 1985) & (df[col_anio] < 2025)]
    conteo_anual = df.groupby(col_anio).size().reset_index(name='Nacimientos')
except Exception as e:
    print(f"Error: {e}")
    exit()

# 2. Modelo basado en la tendencia RECIENTE (Últimos 10 años de GDL)
datos_recientes = conteo_anual[conteo_anual[col_anio] >= 2014]
X_train = datos_recientes[[col_anio]].values
y_train = datos_recientes['Nacimientos'].values

modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones 2025-2026
anios_futuros = np.array([[2025], [2026]])
predicciones = modelo.predict(anios_futuros)

# 3. Configuración de la Gráfica
plt.figure(figsize=(20, 10))

# Graficar Histórico Guadalajara
plt.plot(conteo_anual[col_anio], conteo_anual['Nacimientos'], 
         marker='o', color='#34495e', label='Histórico Guadalajara', linewidth=2.5)

# Graficar Proyección Guadalajara
eje_x_proy = [conteo_anual[col_anio].iloc[-1], 2025, 2026]
eje_y_proy = [conteo_anual['Nacimientos'].iloc[-1], predicciones[0], predicciones[1]]
plt.plot(eje_x_proy, eje_y_proy, marker='s', linestyle='--', color='#e74c3c', label='Proyección GDL', linewidth=2.5)

# --- ETIQUETADO ANUAL VERTICAL ---
todos_x = list(conteo_anual[col_anio]) + [2025, 2026]
todos_y = list(conteo_anual['Nacimientos']) + [float(predicciones[0]), float(predicciones[1])]

for x, y in zip(todos_x, todos_y):
    # Color azul para pasado, rojo para futuro
    color_txt = '#e74c3c' if x > 2024 else '#34495e'
    plt.text(x, y + 800, f'{int(y):,}', 
             ha='center', va='bottom', fontsize=9, 
             fontweight='bold', color=color_txt, rotation=90)

# 4. Estética y Formato
plt.title('Evolución y Proyección de Natalidad: Municipio de Guadalajara (1985 - 2026)', fontsize=18, fontweight='bold')
plt.xlabel('Año de Nacimiento', fontsize=12)
plt.ylabel('Cantidad de Nacimientos Registrados', fontsize=12)
plt.xticks(np.arange(1985, 2027, 1), rotation=45) 
plt.grid(True, linestyle=':', alpha=0.5)
plt.xlim(1984, 2027)
plt.ylim(0, max(todos_y) * 1.25) 

plt.legend(loc='upper right', fontsize=12)
plt.tight_layout()

# Guardar
if not os.path.exists('results'):
    os.makedirs('results')
plt.savefig('results/natalidad_guadalajara_final.png', dpi=300)
print("\n--- PROYECCIÓN GUADALAJARA ---")
print(f"Predicción 2025: {int(predicciones[0]):,} nacimientos")
print(f"Predicción 2026: {int(predicciones[1]):,} nacimientos")
plt.show()