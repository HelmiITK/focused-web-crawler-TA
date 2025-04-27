import pandas as pd
import os

# Folder tempat file CSV berada
folder_name = 'results/unrelevan'

# Membaca semua file CSV dalam folder
csv_files = [os.path.join(folder_name, file) for file in os.listdir(folder_name) if file.endswith('.csv')]

# List untuk menyimpan data dari semua file
dataframes = []

for file in csv_files:
    try:
        print(f"Menggabungkan file: {file}")
        df = pd.read_csv(file)
        dataframes.append(df)
    except Exception as e:
        print(f"Error membaca file {file}: {e}")

# Menggabungkan semua data menjadi satu dataframe
combined_data = pd.concat(dataframes, ignore_index=True)

# Menyimpan ke file CSV baru
output_file = "combined/combined_unrelevan/combined_results_8.csv"
combined_data.to_csv(output_file, index=False)

print(f"File berhasil digabungkan dan disimpan sebagai {output_file}")
