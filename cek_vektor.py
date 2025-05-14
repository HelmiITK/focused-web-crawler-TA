import numpy as np
import pandas as pd

# Load embedding dan dataset judul + label
embeddings = np.load('pre_processing/embedding/combined_embedding.npy')
df = pd.read_csv('pre_processing/label/combined_labeled.csv')

# Pastikan panjang sama
assert embeddings.shape[0] == df.shape[0], "Jumlah embedding dan jumlah data tidak cocok!"

# Contoh lihat data baris ke-0
print("Judul:", df.loc[0, 'Judul'])
print("Label:", df.loc[0, 'Label'])
print("Embedding:", embeddings[0])

# Buat DataFrame gabungan jika mau ditelusuri lebih nyaman
df_combined = df.copy()
df_combined['Embedding'] = [emb for emb in embeddings]

# Lihat beberapa baris pertama
print(df_combined.head())