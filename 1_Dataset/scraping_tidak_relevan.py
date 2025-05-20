import requests
from bs4 import BeautifulSoup
import urllib.parse
import csv
import os
import time
import random

# Meminta input kata kunci dari pengguna
search_query = input("Masukkan kata kunci pencarian: ")

# Mengubah kata kunci pencarian menjadi format URL-encoded
encoded_query = urllib.parse.quote(search_query)

# Membuat URL untuk Google Scholar dengan query pencarian
url = f'https://scholar.google.com/scholar?hl=id&q={encoded_query}'
print(f"Query URL yang akan dikirim: {url}")

# Menambahkan header untuk menyamar sebagai browser
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
# }
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
]
headers = {
    "User-Agent": random.choice(user_agents)
}

# Menambahkan delay sebelum melakukan request
sleep_time = random.uniform(10, 20)
print(f"Menunggu {sleep_time:.2f} detik sebelum mengirim permintaan...")
time.sleep(sleep_time)

# Mengirim permintaan HTTP GET ke URL
response = requests.get(url, headers=headers)

# Mengecek apakah request berhasil
if response.status_code == 200:
    # Mengambil konten halaman
    page_content = response.content
    
    # Membuat objek BeautifulSoup untuk parsing HTML
    soup = BeautifulSoup(page_content, 'html.parser')
    
    # Mencari semua elemen dengan class 'gs_r gs_or gs_scl' yang memuat informasi jurnal
    journals = soup.find_all('div', class_='gs_r gs_or gs_scl')

    # Menentukan folder untuk menyimpan file CSV
    folder = 'results/unrelevan'

    # Mengecek nomor urut file terakhir di folder relevan
    existing_files = os.listdir(folder)
    file_counter = 1

    # Cari file terakhir dengan format hasil_scraping_n.csv
    for file in existing_files:
        if file.startswith('hasil_scraping_') and file.endswith('.csv'):
            try:
                # Ambil nomor urut dari nama file
                num = int(file.split('_')[-1].replace('.csv', ''))
                file_counter = max(file_counter, num + 1)
            except ValueError:
                continue

    # Iterasi melalui setiap jurnal yang ditemukan
    for journal in journals:
        try:
            # Mengambil judul jurnal dari tag 'h3' dengan class 'gs_rt'
            title = journal.find('h3', class_='gs_rt').text.strip()
            author = journal.find('div', class_='gs_a').text.strip()
            # Mengambil abstrak dari tag 'div' dengan class 'gs_rs'
            abstract = journal.find('div', class_='gs_rs').text.strip()

            # Mencetak hasil scraping
            print('-' * 100)
            print(f"Judul: {title}")
            print(f"Penulis: {author}")
            print(f"Abstrak: {abstract}")
            print('-' * 100)

            # Menanyakan apakah ingin menyimpan jurnal ini
            save_option = input("Apakah Anda ingin menyimpan jurnal ini? (y/n)").lower()

            if save_option == 'y':
                    # Membuka file CSV untuk ditulis
                    with open(f'results/unrelevan/hasil_scraping_{file_counter}.csv', mode='w', newline='', encoding='utf-8') as csv_file:
                        fieldnames = ['Judul', 'Penulis', 'Abstrak']
                        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        
                        # Menulis header ke file CSV
                        writer.writeheader()

                        # Menulis data ke file CSV
                        writer.writerow({'Judul': title, 'Penulis': author, 'Abstrak': abstract})

                        print(f"Jurnal disimpan dalam file: hasil_scraping_{file_counter}.csv")
                        print('-' * 100)

                    # Increment file_counter untuk file berikutnya
                    file_counter += 1
            else:
                    print("Jurnal dilewatkan.")

            # Tambahkan jeda antar proses jurnal untuk keamanan ekstra
            time.sleep(random.uniform(5, 10))

        except AttributeError:
            print("Elemen tidak lengkap, melewatkan...")
else:
    print(f"Error: Tidak dapat mengakses halaman. Status code: {response.status_code}")
