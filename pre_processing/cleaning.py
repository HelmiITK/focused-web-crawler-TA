import pandas as pd
import re
import os

# Cleaning Judul saja
def clean_text(text):
  if pd.isna(text):
    return ""
  text = text.lower() # Ubah ke lowercase
  text = re.sub(r'\[.*?\]', '', text)  # hapus [PDF], [DOC], dll
  text = re.sub(r'[^a-z0-9\s]', '', text)  # hapus simbol kecuali huruf dan angka
  text = re.sub(r'\s+', ' ', text).strip()  # hapus spasi berlebih
  return text

# Paths file
current_dir = os.path.dirname(__file__)  # ambil folder path dari cleaning.py
input_path_if = os.path.join(current_dir, '..', 'combined', 'combined_relevan', 'combined_results_6.csv')
input_path_unif = os.path.join(current_dir, '..', 'combined', 'combined_unrelevan', 'combined_results_9.csv')
output_path_if = os.path.join(current_dir, '..', 'pre_processing', 'results_cleaning', 'informatika_clean.csv')
output_path_unif = os.path.join(current_dir, '..', 'pre_processing', 'results_cleaning', 'uninformatika_clean.csv')

# Baca file csv informatika
df_informatika = pd.read_csv(input_path_if)
df_uninformatika = pd.read_csv(input_path_unif)

# Proses cleaning dataset informatika
if 'Judul' in df_informatika.columns:
    df_informatika = df_informatika[['Judul']]
    df_informatika['Judul'] = df_informatika['Judul'].apply(clean_text)
    df_informatika = df_informatika.drop_duplicates()
    df_informatika = df_informatika.reset_index(drop=True)

# Proses cleaning dataset uniformatika
if 'Judul' in df_uninformatika.columns:
    df_uninformatika = df_uninformatika[['Judul']]
    df_uninformatika['Judul'] = df_uninformatika['Judul'].apply(clean_text)
    df_uninformatika = df_uninformatika.drop_duplicates()
    df_uninformatika = df_uninformatika.reset_index(drop=True)
else:
    raise ValueError("Kolom 'Judul' tidak ditemukan di file CSV!")

# Simpan hasil cleaning
os.makedirs(os.path.dirname(output_path_if), exist_ok=True)  # buat folder kalau belum ada
df_informatika.to_csv(output_path_if, index=False)

os.makedirs(os.path.dirname(output_path_unif), exist_ok=True)
df_uninformatika.to_csv(output_path_unif, index=False)

print(f"Sukses! File hasil cleaning disimpan di: {output_path_if}")
print(f"Sukses! File hasil cleaning disimpan di: {output_path_unif}")
