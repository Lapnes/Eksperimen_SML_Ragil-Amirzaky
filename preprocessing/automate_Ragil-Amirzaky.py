"""
Otomatisasi Preprocessing Dataset Credit Scoring - Ragil Amirzaky
==================================================================
Script pembersihan data dan rekayasa fitur (Feature Engineering).
"""

import os
import pandas as pd
import numpy as np

def clean_data(df):
    print("[1/4] Menangani Missing Values (Imputasi Median)...")
    median_income = df['MonthlyIncome'].median()
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(median_income)
    
    median_dependents = df['NumberOfDependents'].median()
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(median_dependents)
    
    print("[2/4] Membersihkan Outlier dan Nilai Ekstrem Kategori...")
    df['RevolvingUtilizationOfUnsecuredLines'] = np.where(
        df['RevolvingUtilizationOfUnsecuredLines'] > 1,
        1.0,
        df['RevolvingUtilizationOfUnsecuredLines']
    )
    
    late_cols = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate'
    ]
    for col in late_cols:
        df[col] = np.where(df[col] >= 96, 5, df[col])
        
    print("[3/4] Rekayasa Fitur (Feature Engineering)...")
    df['TotalLatePayments'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] +
        df['NumberOfTime60-89DaysPastDueNotWorse'] +
        df['NumberOfTimes90DaysLate']
    )
    
    df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    df['IncomePerDependent'] = np.round(df['IncomePerDependent'], 2)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    print("[4/4] Validasi Struktur Data...")
    return df

def main():
    print("=" * 60)
    print("[LOG AUTOMATED PREPROCESSING] Memulai Preprocessing Data")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "..", "credit_scoring_raw", "cs-training.csv")
    output_path = os.path.join(base_dir, "credit_scoring_preprocessing.csv")
    
    if not os.path.exists(input_path):
        print(f"[!] Warning: Raw dataset {input_path} tidak ditemukan!")
        return
        
    df_raw = pd.read_csv(input_path)
    print(f"[*] Input File          : {input_path}")
    print(f"[*] Dimensi Raw Data    : {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")
    
    df_clean = clean_data(df_raw)
    df_clean.to_csv(output_path, index=False)
    
    print(f"[+] Output File         : {output_path}")
    print(f"[+] Dimensi Clean Data  : {df_clean.shape[0]} baris, {df_clean.shape[1]} kolom")
    print(f"[+] Jumlah Column Fitur : {len(df_clean.columns)} kolom")
    print("=" * 60)
    print("[SUCCESS] Automated Preprocessing Selesai!")
    print("=" * 60)

if __name__ == "__main__":
    main()
