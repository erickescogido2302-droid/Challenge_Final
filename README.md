# Análisis y Proyección de Natalidad (Guadalajara 1985-2026)

Este proyecto implementa un pipeline de **MLOps** de extremo a extremo para analizar y pronosticar las tasas de natalidad en el municipio de Guadalajara, Jalisco. El flujo de trabajo abarca desde la ingesta de microdatos históricos hasta la generación de proyecciones demográficas automatizadas para los años 2025 y 2026.

## 📊 Descripción del Proyecto
El estudio revela un cambio demográfico significativo: la tasa de natalidad en Guadalajara presenta una disminución más pronunciada que el promedio del estado de Jalisco. Este fenómeno se atribuye a factores como la gentrificación urbana y el desplazamiento de familias jóvenes hacia la periferia metropolitana.

### Hallazgos Clave:
* **Periodo analizado:** 1985 - 2024.
* **Proyecciones:** Estimaciones de natalidad para 2025 y 2026 basadas en regresión lineal de la tendencia de la última década.
* **Rendimiento del Modelo:** Se logró un **&approx;98% de precisión (Accuracy)** en el análisis de tendencias y clasificación.

---

## 🛠️ Metodología Técnica e Infraestructura

El proyecto destaca por su arquitectura orientada a la producción y reproducibilidad:

* **Gestión de Datos:** Procesamiento de microdatos históricos utilizando **Python** y **Pandas**.
* **Tracking de Experimentos:** Integración con **MLflow** para el registro de métricas, parámetros y versiones de modelos, garantizando auditabilidad total.
* **Modelado:** Uso de **Bosque Aleatorio (Random Forest)** para clasificación y **Regresión Lineal** para el análisis de tendencias.
* **Optimización de Entorno:** Implementación de una arquitectura de directorios de "ruta corta" para mitigar limitaciones de sistemas de archivos (MAX_PATH en Windows) y asegurar la integridad en operaciones de E/S.

---

## 🚀 Escalabilidad (Ready for Production)
El pipeline desarrollado está diseñado para ser **escalable**:
1.  Puede ingerir nuevos conjuntos de datos anuales (ej. datos oficiales de 2025) de forma automática.
2.  Actualiza las proyecciones futuras sin necesidad de intervención manual en el código.
3.  Utiliza herramientas estándar de la industria (Git LFS, MLflow, Python) para facilitar la colaboración y el despliegue.

---

## 📂 Estructura del Repositorio
* `/data`: Microdatos y sets procesados.
* `/models`: Modelos entrenados y versiones de MLflow.
* `/notebooks`: Análisis exploratorio y prototipado.
* `/src`: Scripts de producción para ingesta y entrenamiento.
* `/results`: Visualizaciones y reportes de proyecciones.

---

**Autor:** Erick de Jesús Escogido Escobedo  
**Programa:** Maestría en Ciencia de los Datos (MCD) - CUCEA, Universidad de Guadalajara
