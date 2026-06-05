import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_base_model():
    print("=== STARTING BASE MODELLING WITH MLFLOW AUTOLOG ===")
    
    mlflow.set_experiment("Heart_Disease_Experiment")
    mlflow.autolog()
    
    # 1. Memuat Data Bersih
    train_data = pd.read_csv('preprocessing/heart_preprocessing/train_clean.csv')
    test_data = pd.read_csv('preprocessing/heart_preprocessing/test_clean.csv')
    
    # HOTFIX: Buang baris bolong (NaN) yang bikin eror validate data
    train_data = train_data.dropna()
    test_data = test_data.dropna()
    
    # Memisahkan fitur dan target
    X_train = train_data.drop(columns=['HeartDisease'])
    y_train = train_data['HeartDisease']
    X_test = test_data.drop(columns=['HeartDisease'])
    y_test = test_data['HeartDisease']
    
    # 2. Mulai Run MLflow Tracking
    with mlflow.start_run(run_name="Baseline_Random_Forest"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"-> [SUKSES] Model Dasar Selesai Dilatih. Akurasi Uji: {acc:.4f}")
        print("\n=== CLASSIFICATION REPORT ===")
        print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    train_base_model()
