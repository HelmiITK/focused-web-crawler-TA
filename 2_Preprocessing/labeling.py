import pandas as pd

# Load dataset informatika (relevan)
df_informatika = pd.read_csv('../pre_processing/results_cleaning/informatika_clean.csv')
df_informatika['Label'] = 1  # kasih label 1 untuk jurnal informatika

# Load dataset uninformatika (tidak relevan)
df_uninformatika = pd.read_csv('../pre_processing/results_cleaning/uninformatika_clean.csv')
df_uninformatika['Label'] = 0  # kasih label 0 untuk jurnal bukan informatika

# Gabungkan keduanya
df_combined = pd.concat([df_informatika, df_uninformatika], ignore_index=True)

# Acak barisan data (opsional tapi sangat disarankan sebelum training)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Simpan dataset yang sudah dilabeli
df_combined.to_csv('../pre_processing/label/combined_labeled.csv', index=False)

print("Labeling selesai! File disimpan di 'combined_labeled.csv'")
