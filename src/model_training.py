import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt # <--- NUEVA IMPORTACIÓN

def train_birth_model(data_path):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Natalidad_Guadalajara_MLOps")

    csv_path = os.path.join(data_path, "dataset_limpio.csv")
    if not os.path.exists(csv_path):
        print(f"!!! Error: No se encontró el archivo en {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    median_val = df['ORDEN_PART'].median()
    df['TARGET'] = (df['ORDEN_PART'] > median_val).astype(int)
    
    features = ['EDAD_MADR', 'EDAD_PADR', 'ESCOL_MAD', 'HIJOS_VIVO', 'ANO_NAC']
    X = df[features]
    y = df['TARGET']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    with mlflow.start_run(run_name="RF_Clasificacion_Base"):
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        report = classification_report(y_test, predictions, output_dict=True)
        
        mlflow.set_tag("area_estudio", "Guadalajara")
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", report['accuracy'])
        mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])

        # ==========================================================
        # SECCIÓN DE GRÁFICAS (AQUÍ VAN)
        # ==========================================================
        
        # 1. Gráfica de Importancia de Variables
        plt.figure(figsize=(10, 6))
        importances = pd.Series(model.feature_importances_, index=features)
        importances.sort_values().plot(kind='barh', color='skyblue')
        plt.title("Importancia de Variables - Natalidad")
        plt.tight_layout()
        plt.savefig("importancia_features.png")
        mlflow.log_artifact("importancia_features.png") # Sube a MLflow
        plt.close()

        # 2. Gráfica de Matriz de Confusión
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Blues')
        plt.title("Matriz de Confusión")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png") # Sube a MLflow
        plt.close()
        
        # ==========================================================

        mlflow.sklearn.log_model(model, "model_natalidad")
        
        print("\n" + "="*40)
        print("MÉTRICAS Y GRÁFICAS REGISTRADAS")
        print("="*40)

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_path, "..", "data") 
    train_birth_model(data_folder)