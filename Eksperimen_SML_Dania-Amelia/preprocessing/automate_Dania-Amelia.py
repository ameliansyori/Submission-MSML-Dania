import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def run_automation(input_path='heart.csv', output_dir='preprocessing/heart_preprocessing'):
    print("=== MENJALANKAN OTOMATISASI PREPROCESSING DATASET ===")
    
    # 1. Memuat Data
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} tidak ditemukan!")
        return
    df = pd.read_csv(input_path)
    
    # 2. Memisahkan Fitur dan Target
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']
    
    # 3. Encoding Data Kategorikal
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # 4. Normalisasi Fitur Numerik
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
    
    # 5. Split Train dan Test Set
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Menggabungkan Kembali untuk Disimpan
    train_clean = X_train.copy()
    train_clean['HeartDisease'] = y_train
    
    test_clean = X_test.copy()
    test_clean['HeartDisease'] = y_test
    
    # 7. Membuat Folder Output dan Menyimpan Berkas
    os.makedirs(output_dir, exist_ok=True)
    train_clean.to_csv(f'{output_dir}/train_clean.csv', index=False)
    test_clean.to_csv(f'{output_dir}/test_clean.csv', index=False)
    
    print(f"-> [SUKSES AUTOMATE] Data berhasil dibersihkan secara sistematis!")
    print(f"-> Berkas tersimpan rapi di folder: '{output_dir}/'")

if __name__ == '__main__':
    # Fungsi ini akan berjalan otomatis jika file .py ini dipanggil via terminal
    run_automation()
