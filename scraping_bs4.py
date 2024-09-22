import requests
from bs4 import BeautifulSoup
import urllib.parse
import csv

# Meminta input kata kunci dari pengguna
search_query = input("Masukkan kata kunci pencarian: ")

# Mengubah kata kunci pencarian menjadi format URL-encoded
encoded_query = urllib.parse.quote(search_query)

# Membuat URL untuk Google Scholar dengan query pencarian
url = f'https://scholar.google.com/scholar?hl=id&q={encoded_query}'

# Mengirim permintaan HTTP GET ke URL
response = requests.get(url)

# Mengecek apakah request berhasil
if response.status_code == 200:
    # Mengambil konten halaman
    page_content = response.content
    
    # Membuat objek BeautifulSoup untuk parsing HTML
    soup = BeautifulSoup(page_content, 'html.parser')
    
    # Mencari semua elemen dengan class 'gs_ri' yang memuat informasi jurnal
    # journals = soup.find_all('div', class_='gs_ri')
    journals = soup.find_all('div', class_='gs_r gs_or gs_scl')

    # Membuka file CSV untuk ditulis
    with open('hasil_scraping.csv', mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Judul', 'Penulis', 'Abstrak']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Menulis header ke file CSV
        writer.writeheader()
        
        # Iterasi melalui setiap jurnal yang ditemukan
        for journal in journals:
            # Mengambil judul jurnal dari tag 'a'
            title = journal.find('h3', class_='gs_rt').text.strip()
            
            # Mengambil nama penulis dari tag 'a'
            author = journal.find('div', class_='gs_a').text.strip()
            
            # Mengambil abstrak dari tag 'div' dengan class 'gs_rs'
            abstract = journal.find('div', class_='gs_rs').text.strip()

            # Mencetak hasil scraping
            print(f"Judul: {title}")
            print(f"Penulis: {author}")
            print(f"Abstrak: {abstract}")
            print('-' * 40)
            
            # Menulis data ke file CSV
            writer.writerow({'Judul': title, 'Penulis': author, 'Abstrak': abstract})
else:
    print(f"Error: Tidak dapat mengakses halaman. Status code: {response.status_code}")
