import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_tuning_model():
    print("=== STARTING HYPERPARAMETER TUNING WITH MANUAL LOGGING ===")
    
    mlflow.set_experiment("Heart_Disease_Experiment")
    
    # 1. Memuat Data Bersih
    train_data = pd.read_csv('heart_preprocessing/train_clean.csv')
    test_data = pd.read_csv('heart_preprocessing/test_clean.csv')
    
    X_train = train_data.drop(columns=['HeartDisease'])
    y_train = train_data['HeartDisease']
    X_test = test_data.drop(columns=['HeartDisease'])
    y_test = test_data['HeartDisease']
    
    # 2. Menentukan Parameter yang Mau Diuji (Grid Search)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    
    base_rf = RandomForestClassifier(random_state=42)
    
    # Mencari kombinasi terbaik menggunakan GridSearchCV
    grid_search = GridSearchCV(estimator=base_rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # 3. Manual Logging ke MLflow untuk Model Terbaik (Skilled Requirement)
    with mlflow.start_run(run_name="Tuned_Random_Forest"):
        # Log parameter terbaik secara manual
        for param_name, param_val in best_params.items():
            mlflow.log_param(param_name, param_val)
        
        # Melakukan prediksi menggunakan model terbaik
        y_pred = best_model.predict(X_test)
        
        # Menghitung berbagai metriks evaluasi
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metriks secara manual ke MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # Menyimpan file model final secara fisik sebagai artefak MLflow
        mlflow.sklearn.log_model(best_model, "best_rf_model")
        
        print(f"-> [SUKSES TUNING] Parameter Terbaik: {best_params}")
        print(f"-> Metriks Terbaik Tercatat Manual - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

if __name__ == '__main__':
    train_tuning_model()
