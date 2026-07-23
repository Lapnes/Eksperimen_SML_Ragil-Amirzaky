"""
Generate Credit Scoring Dataset - Ragil Amirzaky
================================================
Script pembuatan dataset sintetis credit scoring.
"""

import os
import pandas as pd
import numpy as np

def generate_credit_scoring_dataset(n_samples=10000, random_state=42):
    print("=" * 60)
    print("[LOG DATASET GENERATOR] Memulai Pembuatan Dataset Credit Scoring")
    print("=" * 60)
    
    np.random.seed(random_state)
    
    print(f"[*] Jumlah sampel : {n_samples:,} baris")
    print("[*] Membuat fitur numerik dan kategorikal...")
    
    age = np.random.normal(52, 14, n_samples).astype(int)
    age = np.clip(age, 21, 90)
    
    monthly_income = np.random.lognormal(8.5, 0.8, n_samples).astype(int)
    monthly_income = np.clip(monthly_income, 0, 50000)
    
    revolving_utilization = np.random.beta(2, 5, n_samples)
    outlier_mask = np.random.random(n_samples) < 0.05
    revolving_utilization[outlier_mask] = np.random.uniform(1, 13, outlier_mask.sum())
    
    debt_ratio = np.random.lognormal(-0.5, 1.2, n_samples)
    debt_ratio = np.clip(debt_ratio, 0, 10)
    
    num_open_credit = np.random.poisson(8, n_samples)
    num_real_estate = np.random.poisson(1, n_samples)
    num_dependents = np.random.poisson(0.7, n_samples)
    
    num_30_59_late = np.zeros(n_samples, dtype=int)
    mask_30 = np.random.random(n_samples) < 0.15
    num_30_59_late[mask_30] = np.random.choice([1, 2, 3, 4, 5, 98], mask_30.sum(), 
                                                 p=[0.5, 0.2, 0.1, 0.05, 0.05, 0.1])
    
    num_60_89_late = np.zeros(n_samples, dtype=int)
    mask_60 = np.random.random(n_samples) < 0.08
    num_60_89_late[mask_60] = np.random.choice([1, 2, 3, 96, 98], mask_60.sum(),
                                                 p=[0.5, 0.2, 0.1, 0.1, 0.1])
    
    num_90_late = np.zeros(n_samples, dtype=int)
    mask_90 = np.random.random(n_samples) < 0.06
    num_90_late[mask_90] = np.random.choice([1, 2, 3, 96, 98], mask_90.sum(),
                                              p=[0.4, 0.2, 0.15, 0.1, 0.15])
    
    log_odds = (
        -3.0
        + 0.5 * (revolving_utilization > 1).astype(float)
        + 0.3 * (debt_ratio > 1).astype(float)
        + 0.8 * (num_30_59_late > 0).astype(float)
        + 1.2 * (num_90_late > 0).astype(float)
        - 0.02 * (age - 50)
        - 0.0001 * monthly_income
        + 0.5 * (num_60_89_late > 0).astype(float)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    target = (np.random.random(n_samples) < prob).astype(int)
    
    monthly_income_with_na = monthly_income.astype(float)
    na_income_mask = np.random.random(n_samples) < 0.20
    monthly_income_with_na[na_income_mask] = np.nan
    
    num_dependents_with_na = num_dependents.astype(float)
    na_dep_mask = np.random.random(n_samples) < 0.03
    num_dependents_with_na[na_dep_mask] = np.nan
    
    df = pd.DataFrame({
        'Unnamed: 0': range(1, n_samples + 1),
        'SeriousDlqin2yrs': target,
        'RevolvingUtilizationOfUnsecuredLines': np.round(revolving_utilization, 6),
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': num_30_59_late,
        'DebtRatio': np.round(debt_ratio, 6),
        'MonthlyIncome': monthly_income_with_na,
        'NumberOfOpenCreditLinesAndLoans': num_open_credit,
        'NumberOfTimes90DaysLate': num_90_late,
        'NumberRealEstateLoansOrLines': num_real_estate,
        'NumberOfTime60-89DaysPastDueNotWorse': num_60_89_late,
        'NumberOfDependents': num_dependents_with_na
    })
    
    print("[*] Generasi target variable selesai.")
    return df

if __name__ == "__main__":
    df = generate_credit_scoring_dataset(n_samples=10000)
    
    output_dir = os.path.join(os.path.dirname(__file__), "credit_scoring_raw")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "cs-training.csv")
    df.to_csv(output_path, index=False)
    
    print(f"[+] Output File        : {output_path}")
    print(f"[+] Dimensi Dataset    : {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"[+] Distribusi Target  :\n{df['SeriousDlqin2yrs'].value_counts().to_string()}")
    print("=" * 60)
    print("[SUCCESS] Dataset raw berhasil dibuat!")
    print("=" * 60)
