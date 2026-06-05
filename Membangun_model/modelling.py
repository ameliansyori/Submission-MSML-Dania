import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_base_model():
    print("=== STARTING BASE MODELLING WITH MLFLOW AUTOLOG ===")
    
    # 1. Mengaktifkan MLflow Autolog (Mencatat semua parameter & metriks otomatis)
    mlflow.set_experiment("Heart_Disease_Experiment")
    mlflow.autolog()
    
    # 2. Memuat Data Bersih Hasil Preprocessing Kriteria 1
    # Sesuaikan path ini dengan posisi folder 'heart_preprocessing' lo nanti
    train_data = pd.read_csv('heart_preprocessing/train_clean.csv')
    test_data = pd.read_csv('heart_preprocessing/test_clean.csv')
    
    # Memisahkan fitur dan target
    X_train = train_data.drop(columns=['HeartDisease'])
    y_train = train_data['HeartDisease']
    X_test = test_data.drop(columns=['HeartDisease'])
    y_test = test_data['HeartDisease']
    
    # 3. Mulai Run MLflow Tracking
    with mlflow.start_run(run_name="Baseline_Random_Forest"):
        # Inisialisasi model dengan parameter default
        model = RandomForestClassifier(random_state=42)
        
        # Melatih Model (Autolog akan merekam proses ini secara otomatis)
        model.fit(X_train, y_train)
        
        # Prediksi data uji
        y_pred = model.predict(X_test)
        
        # Evaluasi manual tambahan untuk log cetak di terminal
        acc = accuracy_score(y_test, y_pred)
        print(f"-> [SUKSES] Model Dasar Selesai Dilatih. Akurasi Uji: {acc:.4f}")
        print("\n=== CLASSIFICATION REPORT ===")
        print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    train_base_model()
