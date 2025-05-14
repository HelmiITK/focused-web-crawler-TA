# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # Buat folder jika belum ada
# os.makedirs("metrix_evaluation", exist_ok=True)

# # ========== 1. Load Dataset Sebelum Cleaning ==========
# raw_relevan_path = "D:/Semester 8/TA/scraping-with-beautifulsoup/combined/combined_relevan/combined_results_6.csv"
# raw_unrelevan_path = "D:/Semester 8/TA/scraping-with-beautifulsoup/combined/combined_unrelevan/combined_results_9.csv"

# raw_relevan = pd.read_csv(raw_relevan_path)
# raw_unrelevan = pd.read_csv(raw_unrelevan_path)

# # ========== 2. Load Dataset Setelah Cleaning ==========
# clean_informatika_path = "D:/Semester 8/TA/scraping-with-beautifulsoup/pre_processing/results_cleaning/informatika_clean.csv"
# clean_uninformatika_path = "D:/Semester 8/TA/scraping-with-beautifulsoup/pre_processing/results_cleaning/uninformatika_clean.csv"

# clean_relevan = pd.read_csv(clean_informatika_path)
# clean_unrelevan = pd.read_csv(clean_uninformatika_path)

# # ========== 3. Hitung Jumlah ==========
# jumlah_raw = [len(raw_relevan), len(raw_unrelevan)]
# jumlah_clean = [len(clean_relevan), len(clean_unrelevan)]

# # ========== 4. Visualisasi ==========
# labels = ['Relevan (Informatika)', 'Tidak Relevan (Uninformatika)']
# x = range(len(labels))

# plt.figure(figsize=(8, 6))
# bar1 = plt.bar(x, jumlah_raw, width=0.4, label='Sebelum Cleaning', align='center', color='#2196F3')
# bar2 = plt.bar([i + 0.4 for i in x], jumlah_clean, width=0.4, label='Sesudah Cleaning', align='center', color='#4CAF50')

# plt.xticks([i + 0.2 for i in x], labels)
# plt.ylabel('Jumlah Data')
# plt.title('Perbandingan Jumlah Data Sebelum dan Sesudah Cleaning')
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.6)

# # Tambahkan label angka di atas bar
# for bars in [bar1, bar2]:
#     for bar in bars:
#         yval = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom', fontsize=10)

# # Simpan hasil visualisasi
# plt.tight_layout()
# # plt.savefig('metrix_evaluation/perbandingan_data_cleaning.png')
# plt.show()


#==========================================================================================================================================

# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # Buat folder jika belum ada
# os.makedirs("metrix_evaluation", exist_ok=True)

# # ========== Load Dataset Sebelum Cleaning ==========
# raw_relevan_path = "D:/Semester 8/TA/scraping-with-beautifulsoup/combined/combined_relevan/combined_results_6.csv"
# raw_unrelevan_path = "D:/Semester 8/TA/scraping-with-beautifulsoup/combined/combined_unrelevan/combined_results_9.csv"

# raw_relevan = pd.read_csv(raw_relevan_path)
# raw_unrelevan = pd.read_csv(raw_unrelevan_path)

# # ========== Hitung Jumlah ==========
# jumlah_raw = [len(raw_relevan), len(raw_unrelevan)]
# labels = ['Relevan (Informatika)', 'Tidak Relevan (Uninformatika)']
# colors = ['#2196F3', '#4CAF50']

# # ========== Visualisasi ==========
# plt.figure(figsize=(6, 6))
# bars = plt.bar(labels, jumlah_raw, color=colors)

# plt.ylabel('Jumlah Data')
# plt.title('Jumlah Dataset Judul Jurnal')
# plt.grid(axis='y', linestyle='--', alpha=0.6)

# # Tambahkan label angka di atas bar
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom', fontsize=10)

# plt.tight_layout()
# # plt.savefig('metrix_evaluation/jumlah_data_sebelum_cleaning.png')
# plt.show()

#==========================================================================================================================================

# import pandas as pd
# file_path = "D:/Semester 8/TA/scraping-with-beautifulsoup/combined/combined_relevan/combined_results_6.csv"

# # Membaca file CSV
# data = pd.read_csv(file_path)
# # print("Informasi Dataset:")
# print("\nDataset judul infotmatika:")
# print(data.info())

# # Menampilkan beberapa baris pertama
# print(data)


#==========================================================================================================================================

# import pandas as pd

# # Path file
# raw_path = "D:/Semester 8/TA/scraping-with-beautifulsoup/combined/combined_relevan/combined_results_6.csv"
# clean_path = "D:/Semester 8/TA/scraping-with-beautifulsoup/pre_processing/results_cleaning/informatika_clean.csv"

# # Load data
# raw_df = pd.read_csv(raw_path)
# clean_df = pd.read_csv(clean_path)

# # Index yang ingin dibandingkan
# index = 0  

# # Cek panjang dataset untuk menghindari IndexError
# if index < len(raw_df) and index < len(clean_df):
#     print(f"Index ke-{index}")
#     print("-" * 50)
#     print("Sebelum Cleaning :")
#     print(raw_df.iloc[index]['Judul'])  # Ganti 'title' jika nama kolom berbeda
#     print("\nSesudah Cleaning :")
#     print(clean_df.iloc[index]['Judul'])  # Ganti 'title' jika nama kolom berbeda
# else:
#     print("Index melebihi panjang dataset.")


#==========================================================================================================================================

import pandas as pd

# Path file setelah dilakukan labeling
labeled_path = "D:/Semester 8/TA/focused-web-crawler/pre_processing/label/combined_labeled.csv"

# Load dataset
df = pd.read_csv(labeled_path)

# Tampilkan 5 data pertama dan 5 data terakhir
print(f"Total Dataset:{len(df)}")
print(df[['Judul', 'Label']].iloc[0:10])  
print(df[['Judul', 'Label']].tail(10))


