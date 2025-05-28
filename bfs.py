import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import numpy as np
import heapq
import json
import joblib
import os
import queue
import time
import torch
from transformers import BertTokenizer, BertModel

# =============================
# Load Model & Tokenizer
# =============================
knn = joblib.load('3_Training_KNN/save_model_knn/knn_model_k4.pkl')
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = BertModel.from_pretrained('indobenchmark/indobert-base-p1')

# =============================
# Konfigurasi
# =============================
SEED_FILE_BFS = 'seed_urls_percobaan_1.txt'
SEED_FILE_SHARK = 'seed_urls_percobaan_2.txt'
MAX_RESULTS = 100

# Shark Search Parameters
delta = 0.7
beta = 0.6
gamma = 0.6
threshold_score = 0.4

# =============================
# Fungsi BERT, KNN, Cosine Sim
# =============================
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy().flatten()

def predict_relevance(Judul):
    vector = embed_text(Judul).reshape(1, -1)
    return knn.predict(vector)[0]

def cosine_sim(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm if norm != 0 else 0.0

# =============================
# Fungsi Validasi & File
# =============================
def is_valid_url(url):
    return url.startswith("http") and not url.endswith(".pdf")

def read_seed_urls(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def save_results_to_json(data, filename):
    folder = 'crawler_outputs'
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_raw_html(soup, index):
    folder = 'downloaded_pages'
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f'page_{index}.html')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(str(soup))

def is_valid_journal(title: str, link: str) -> bool:
    if not title or len(title.strip().split()) < 4:
        return False

    title_lower = title.strip().lower()
    link_lower = link.strip().lower()
    stop_keywords = [
        "home", "log in", "search", "editorial team", "author guidelines", "editorial policies",
        "online submission", "open access", "statcounter", "about", "review", "panduan penulis",
        "proses review", "copyright", "license", "crossmark", "cite", "submission", "hak cipta dan lisensi"
    ]
    if any(k in title_lower for k in stop_keywords):
        return False
    if any(k in link_lower for k in stop_keywords):
        return False
    return True

# =============================
# BFS Crawler
# =============================
def crawl_page(url, visited_urls, urls_to_visit, relevant_docs, irrelevant_docs):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.find('title')
        title_text = title_tag.get_text().strip() if title_tag else ''
        print(f"  Judul: {title_text}")

        if title_text:
            prediction = predict_relevance(title_text)
            if prediction == 1:
                print("  ✅ Relevan")
                relevant_docs.append(url)
            else:
                print("  ❌ Tidak relevan")
                irrelevant_docs.append(url)
        else:
            print("  ⚠️ Tidak ada judul")
            irrelevant_docs.append(url)

        for link in soup.find_all('a', href=True):
            absolute_url = urljoin(url, link['href'])
            parsed = urlparse(absolute_url)
            if parsed.scheme in ['http', 'https']:
                if absolute_url not in visited_urls and absolute_url not in urls_to_visit.queue:
                    urls_to_visit.put(absolute_url)

    except requests.exceptions.RequestException as e:
        print(f"  ⚠️ Gagal mengambil URL: {url} - {e}")

    return visited_urls, urls_to_visit, relevant_docs, irrelevant_docs

def bfs_crawler(seed_urls, max_depth):
    visited_urls = set()
    urls_to_visit = queue.Queue()
    for url in seed_urls:
        urls_to_visit.put(url)

    relevant_docs = []
    irrelevant_docs = []
    total_visited = 0

    while not urls_to_visit.empty() and total_visited < max_depth:
        current_url = urls_to_visit.get()
        if current_url in visited_urls:
            continue

        print(f"\n[{total_visited+1}] Mengunjungi: {current_url}")
        visited_urls.add(current_url)

        visited_urls, urls_to_visit, relevant_docs, irrelevant_docs = crawl_page(
            current_url, visited_urls, urls_to_visit, relevant_docs, irrelevant_docs
        )

        total_visited += 1

    print("\n--- Ringkasan BFS ---")
    print(f"Total dikunjungi       : {total_visited}")
    print(f"Dokumen relevan        : {len(relevant_docs)}")
    print(f"Dokumen tidak relevan  : {len(irrelevant_docs)}")
    if total_visited > 0:
        print(f"Harvest Rate           : {len(relevant_docs) / total_visited:.4f}")
    else:
        print("Harvest Rate           : 0.0000")

# =============================
# Shark Search Crawler (Potongan Awal)
# =============================
def shark_search_crawler(query, max_depth=3, size=100, time_limit=400, width=10):
    frontier = []
    heap_counter = 0
    visited = set()
    relevant_results = []
    query_vec = embed_text(query)

    total_documents_visited = 0
    start_time = time.time()

    for seed_url in read_seed_urls(SEED_FILE_SHARK):
        inherited = 1.0
        anchor = 1.0
        context = 1.0
        neighborhood_score = beta * anchor + (1 - beta) * context
        potential_score = gamma * inherited + (1 - gamma) * neighborhood_score
        heapq.heappush(frontier, (-potential_score, heap_counter, {
            'url': seed_url,
            'depth': 0,
            'inherited_score': inherited
        }))
        heap_counter += 1

    print(f"[INFO] Memulai crawler dengan query: {query}")

    while frontier and len(relevant_results) < MAX_RESULTS:
        if total_documents_visited >= size:
            print("[STOP] Batas size tercapai.")
            break
        if (time.time() - start_time) > time_limit:
            print("[STOP] Batas waktu tercapai.")
            break

        _, _, node = heapq.heappop(frontier)
        url = node['url']
        depth = node['depth']
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        print(f"\n[DEPTH {depth}] Mengunjungi: {url}")
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            save_raw_html(soup, len(visited))
            total_documents_visited += 1

            # Proses relevansi
            title = soup.title.string.strip() if soup.title else ''
            if is_valid_journal(title, url):
                prediction = predict_relevance(title)
                if prediction == 1:
                    print("  ✅ Relevan")
                    relevant_results.append({'url': url, 'title': title})
                else:
                    print("  ❌ Tidak relevan")

            # Ekstraksi link
            for a in soup.find_all('a', href=True):
                anchor_text = a.get_text(strip=True)
                href = a.get('href')
                next_url = urljoin(url, href)
                if not is_valid_url(next_url) or next_url in visited:
                    continue
                heapq.heappush(frontier, (-0.5, heap_counter, {
                    'url': next_url,
                    'depth': depth + 1,
                    'inherited_score': 0.5
                }))
                heap_counter += 1

        except Exception as e:
            print(f"[ERROR] Tidak dapat mengakses {url}: {e}")
            continue

    print(f"\n--- Ringkasan Shark Search ---")
    print(f"Total dikunjungi       : {total_documents_visited}")
    print(f"Dokumen relevan        : {len(relevant_results)}")
    if total_documents_visited > 0:
        print(f"Harvest Rate           : {len(relevant_results) / total_documents_visited:.4f}")

# =============================
# Main
# =============================
if __name__ == '__main__':
    mode = input("Pilih mode crawler (bfs/shark): ").strip().lower()
    if mode == 'bfs':
        seed_urls = read_seed_urls(SEED_FILE_BFS)
        bfs_crawler(seed_urls, max_depth=100)
    elif mode == 'shark':
        query = input("Masukkan query pencarian: ")
        shark_search_crawler(query, max_depth=3, size=100)
    else:
        print("Mode tidak dikenali.")
