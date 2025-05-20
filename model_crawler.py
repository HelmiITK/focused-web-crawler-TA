# import requests
# from bs4 import BeautifulSoup
# import numpy as np
# import json
# from urllib.parse import urljoin
# from sklearn.neighbors import KNeighborsClassifier
# import joblib
# from transformers import BertTokenizer, BertModel
# import torch
# import os
# import heapq

# knn = joblib.load('save_model_knn/knn_model_k4.pkl')
# tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
# model = BertModel.from_pretrained('indobenchmark/indobert-base-p1')

# # Parameter Shark Search
# delta = 0.8 # δ 0.1 - 1.0
# beta = 0.8 # β 0.1 - 5.0
# gamma = 0.7 # γ 0.5 - 2.0
# MAX_RESULTS = 100  # batas jumlah dokumen yang ingin dikumpulkan

# def embed_text(text):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).numpy().flatten()

# def predict_relevance(Judul):
#     vector = embed_text(Judul).reshape(1, -1)
#     return knn.predict(vector)[0]

# def cosine_sim(vec1, vec2):
#     dot = np.dot(vec1, vec2)
#     norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
#     return dot / norm if norm != 0 else 0.0

# def is_valid_url(url):
#     # Hindari file PDF dan domain yang mencurigakan
#     return url.startswith("http") and not url.endswith(".pdf")

# def shark_search_crawler(query, max_depth=5):
#     frontier = []
#     heap_counter = 0
#     visited = set()
#     results = []

#     total_documents_visited = 0         # Jumlah halaman web dikunjungi
#     total_documents_evaluated = 0       # Jumlah dokumen (link) yang dievaluasi relevansinya

#     seed_url = f"https://scholar.google.com/scholar?q={query}"
#     seed_inherited_score = 1.0
#     query_vec = embed_text(query)

#     seed_anchor_score = 1.0
#     seed_context_score = 1.0
#     neighborhood_score = beta * seed_anchor_score + (1 - beta) * seed_context_score
#     potential_score = gamma * seed_inherited_score + (1 - gamma) * neighborhood_score

#     heapq.heappush(frontier, (-potential_score, heap_counter, {
#         'url': seed_url,
#         'depth': 0,
#         'inherited_score': seed_inherited_score
#     }))
#     heap_counter += 1

#     print(f"[INFO] Memulai crawler dengan query: {query}")

#     while frontier and len(results) < MAX_RESULTS:
#         _, _, node = heapq.heappop(frontier)
#         url = node['url']
#         inherited_score = node['inherited_score']
#         depth = node['depth']

#         if url in visited or depth > max_depth:
#             continue
#         visited.add(url)

#         print(f"\n[DEPTH {depth}] Mengunjungi: {url}")
#         try:
#             response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
#             soup = BeautifulSoup(response.content, 'html.parser')
#             total_documents_visited += 1
#         except Exception as e:
#             print(f"[ERROR] Gagal mengakses {url}: {e}")
#             continue

#         links = soup.select('h3 a')
#         print(f"[INFO] Ditemukan {len(links)} link di halaman ini.")

#         for a in links:
#             title = a.get_text(strip=True)
#             anchor_href = a.get('href')
#             anchor_url = urljoin(url, anchor_href)

#             if not is_valid_url(anchor_url):
#                 print(f"[SKIP] Lewati URL tidak valid atau PDF: {anchor_url}")
#                 continue

#             total_documents_evaluated += 1  # ← Hitung semua link yang diperiksa

#             try:
#                 anchor_vec = embed_text(title)
#                 anchor_score = cosine_sim(query_vec, anchor_vec)

#                 if anchor_score == 0:
#                     context_text = soup.get_text()[:500]
#                     anchor_context_score = cosine_sim(query_vec, embed_text(context_text))
#                 else:
#                     anchor_context_score = 1.0

#                 neighborhood_score = beta * anchor_score + (1 - beta) * anchor_context_score
#                 child_inherited_score = delta * anchor_score if anchor_score > 0 else delta * inherited_score
#                 potential_score = gamma * child_inherited_score + (1 - gamma) * neighborhood_score

#                 is_relevant = predict_relevance(title)
#                 print(f"[CHECK] '{title}' | Anchor Score: {anchor_score:.3f} | Relevan: {is_relevant}")
                
#                 if is_relevant == 1:
#                     results.append({
#                         'id': len(results) + 1,
#                         'title': title, 
#                         'link': anchor_url,
#                         'anchor_score': float(f"{anchor_score:.4f}")
#                     })
#                     print(f"[RELEVAN] +1 → {len(results)} total")

#                 heapq.heappush(frontier, (-potential_score, heap_counter, {
#                     'url': anchor_url,
#                     'depth': depth + 1,
#                     'inherited_score': child_inherited_score
#                 }))
#                 heap_counter += 1

#             except Exception as e:
#                 print(f"[ERROR] Gagal proses link: {e}")
#                 continue
    
#     # Perhitungan harvest rate
#     harvest_rate = len(results) / total_documents_evaluated if total_documents_evaluated > 0 else 0.0
#     print(f"\n[INFO] Total halaman web dikunjungi: {total_documents_visited}")
#     print(f"[INFO] Total dokumen dievaluasi: {total_documents_evaluated}")
#     print(f"[INFO] Total dokumen relevan: {len(results)}")
#     print(f"[INFO] Harvest Rate (HR): {harvest_rate:.4f}")

#     return results

# def save_results_to_json(data, filename='hasil_crawling_log_25.json'):
#     folder = 'crawler_outputs'
#     os.makedirs(folder, exist_ok=True) 
#     filepath = os.path.join(folder, filename)

#     with open(filepath, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)

# if __name__ == "__main__":
#     query = "Focused crawler untuk judul jurnal informatika"
#     results = shark_search_crawler(query)
#     save_results_to_json(results)
#     print(f"\n[SELESAI] Total dokumen relevan ditemukan: {len(results)}")


#=================================================================================================================================================================


import requests
from bs4 import BeautifulSoup
import numpy as np
import json
from urllib.parse import urljoin
from sklearn.neighbors import KNeighborsClassifier
import joblib
from transformers import BertTokenizer, BertModel
import torch
import os
import heapq

# Load KNN dan BERT
knn = joblib.load('3_Training_KNN/save_model_knn/knn_model_k4.pkl')
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = BertModel.from_pretrained('indobenchmark/indobert-base-p1')

# Parameter Shark Search
delta = 0.7
beta = 0.6
gamma = 0.8
MAX_RESULTS = 100

# BACA SEED URL DARI FILE
# SEED_FILE = 'seed_urls.txt'
SEED_FILE = 'seed_urls_dengan_judul.txt'

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

def is_valid_url(url):
    return url.startswith("http") and not url.endswith(".pdf")

def read_seed_urls():
    with open(SEED_FILE, 'r', encoding='utf-8') as f:
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

def shark_search_crawler(query, max_depth=5):
    frontier = []
    heap_counter = 0
    visited = set()
    relevant_results = []
    query_vec = embed_text(query)

    total_documents_visited = 0
    total_documents_evaluated = 0

    for seed_url in read_seed_urls():
        seed_inherited_score = 1.0
        seed_anchor_score = 1.0
        seed_context_score = 1.0
        neighborhood_score = beta * seed_anchor_score + (1 - beta) * seed_context_score
        potential_score = gamma * seed_inherited_score + (1 - gamma) * neighborhood_score

        heapq.heappush(frontier, (-potential_score, heap_counter, {
            'url': seed_url,
            'depth': 0,
            'inherited_score': seed_inherited_score
        }))
        heap_counter += 1

    print(f"[INFO] Memulai crawler dengan query: {query}")

    while frontier and len(relevant_results) < MAX_RESULTS:
        _, _, node = heapq.heappop(frontier)
        url = node['url']
        inherited_score = node['inherited_score']
        depth = node['depth']

        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        print(f"\n[DEPTH {depth}] Mengunjungi: {url}")
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            total_documents_visited += 1

            # Simpan halaman web yang telah diunduh
            save_raw_html(soup, len(visited))

        except Exception as e:
            print(f"[ERROR] Gagal mengakses {url}: {e}")
            continue

        links = soup.find_all('a')
        print(f"[INFO] Ditemukan {len(links)} link di halaman ini.")

        for a in links:
            title = a.get_text(strip=True)
            anchor_href = a.get('href')
            anchor_url = urljoin(url, anchor_href) if anchor_href else None

            if not anchor_url or not is_valid_url(anchor_url):
                continue

            total_documents_evaluated += 1

            try:
                anchor_vec = embed_text(title)
                anchor_score = cosine_sim(query_vec, anchor_vec)

                if anchor_score == 0:
                    context_text = soup.get_text()[:500]
                    anchor_context_score = cosine_sim(query_vec, embed_text(context_text))
                else:
                    anchor_context_score = 1.0

                neighborhood_score = beta * anchor_score + (1 - beta) * anchor_context_score
                child_inherited_score = delta * anchor_score if anchor_score > 0 else delta * inherited_score
                potential_score = gamma * child_inherited_score + (1 - gamma) * neighborhood_score

                is_relevant = predict_relevance(title)
                print(f"[CHECK] '{title}' | Anchor Score: {anchor_score:.3f} | Relevan: {is_relevant}")

                hasil = {
                    'id': total_documents_evaluated,
                    'title': title,
                    'link': anchor_url,
                    'anchor_score': float(f"{anchor_score:.4f}"),
                    'context_score': float(f"{anchor_context_score:.4f}"),
                    'inherited_score': float(f"{child_inherited_score:.4f}"),
                    'neighborhood_score': float(f"{neighborhood_score:.4f}"),
                    'potential_score': float(f"{potential_score:.4f}"),
                    'relevance': int(is_relevant)
                }

                if is_relevant == 1:
                    relevant_results.append(hasil)
                    print(f"[RELEVAN] +1 → {len(relevant_results)} total")

                if depth < max_depth:
                    heapq.heappush(frontier, (-potential_score, heap_counter, {
                        'url': anchor_url,
                        'depth': depth + 1,
                        'inherited_score': child_inherited_score
                    }))
                    heap_counter += 1

            except Exception as e:
                print(f"[ERROR] Gagal proses link: {e}")
                continue

    harvest_rate = len(relevant_results) / total_documents_evaluated if total_documents_evaluated > 0 else 0.0
    print(f"\n[INFO] Total halaman web dikunjungi: {total_documents_visited}")
    print(f"[INFO] Total dokumen dievaluasi: {total_documents_evaluated}")
    print(f"[INFO] Total dokumen relevan: {len(relevant_results)}")
    print(f"[INFO] Harvest Rate (HR): {harvest_rate:.4f}")

    #save hasil crawling
    save_results_to_json(relevant_results, 'hasil_crawling_log_final_dengan_judul.json')

    return relevant_results

if __name__ == "__main__":
    query = "Focused crawler untuk judul jurnal informatika"
    # query = "Deteksi penlanggaran lalulintas menggunakan model Yolo"
    # query = "Pengembangan Sistem Identifikasi Fakta Dan Tidak Fakta Berita di Media Informasi Berbahasa Indonesia"
    results = shark_search_crawler(query)
    print(f"\n[SELESAI] Total dokumen relevan ditemukan: {len(results)}")
