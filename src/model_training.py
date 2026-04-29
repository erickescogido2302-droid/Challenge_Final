import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import os

def train_birth_model(data_path):
    # 1. Cargar el dataset limpio que generamos
    df = pd.read_csv(os.path.join(data_path, "dataset_limpio.csv"))
    
    # 2. Crear la Variable Objetivo (Clasificación)
    # Definimos 'Alta' (1) si el orden de parto es mayor a la mediana, 'Baja' (0) si no.
    # Nota: Puedes ajustar esta lógica según tu tesis (ej. nacimientos por año).
    median_val = df['ORDEN_PART'].median()
    df['TARGET'] = (df['ORDEN_PART'] > median_val).astype(int)
    
    # 3. Selección de Características (Features)
    # Usaremos variables clave: Edad Madre, Edad Padre, Escolaridad, etc.
    features = ['EDAD_MADR', 'EDAD_PADR', 'ESCOL_MAD', 'HIJOS_VIVO', 'ANO_NAC']
    X = df[features]
    y = df['TARGET']
    
    # 4. División de datos (Train/Test Split - Requisito del Challenge)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Normalización (Requisito del Challenge)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. MLOps con MLflow (Stage 3)
    mlflow.set_experiment("Natalidad_Jalisco_Experiment")
    
    with mlflow.start_run():
        # Entrenar Modelo
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predicciones
        predictions = model.predict(X_test_scaled)
        
        # Métricas (Requisito: Precision, Recall, F1)
        report = classification_report(y_test, predictions, output_dict=True)
        
        # Registrar parámetros y métricas en MLflow
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("accuracy", report['accuracy'])
        mlflow.log_metric("precision", report['weighted avg']['precision'])
        mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])
        
        # Guardar el modelo (Artifact)
        mlflow.sklearn.log_model(model, "model_natalidad")
        
        print("\n" + "="*30)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*30)
        print(f"Accuracy: {report['accuracy']:.2f}")
        print("\nMatriz de Confusión:\n", confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    train_birth_model("data/")