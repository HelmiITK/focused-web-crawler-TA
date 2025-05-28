import requests
from bs4 import BeautifulSoup
import numpy as np
import json
from urllib.parse import urljoin, urlparse
from sklearn.neighbors import KNeighborsClassifier
import joblib
from transformers import BertTokenizer, BertModel
import torch
import os
import heapq
import time
import queue
import re

# Load KNN dan BERT
knn = joblib.load('3_Training_KNN/save_model_knn/knn_model_k4.pkl')
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = BertModel.from_pretrained('indobenchmark/indobert-base-p1')

# Parameter Shark Search
delta = 0.7 # δ  
beta = 0.6  # β
gamma = 0.6 # γ
# MAX_RESULTS = 100

threshold_score = 0.4

# BACA SEED URL DARI FILE [seed urls awal untuk dilakuakn crawl oleh shark search]
SEED_FILE = 'seed_urls_percobaan_1.txt'

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
    folder = 'downloaded_pages_percobaan_1_100'
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f'page_{index}.html')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(str(soup))

def is_valid_url(url):
    return url.startswith("http") and not url.endswith(".pdf")

# Fungsi untuk menangani batasan masalah pada model KNN untuk menyimpan ke results
def is_valid_journal(title: str, link: str) -> bool:
    if not title or len(title.strip().split()) < 4:
        return False
    
    title_lower = title.strip().lower()
    link_lower = link.strip().lower()

    # Kata-kata umum non-jurnal yang harus disaring
    stop_keywords = [
        "home", "log in", "search", "editorial team", "author guidelines", "editorial policies",
        "online submission", "open access", "statcounter", "about", "review", "panduan penulis",
        "proses review", "copyright", "license", "crossmark", "cite", "submission", "hak cipta dan lisensi"
    ]
    
    # Domain atau kata kunci umum yang biasanya bukan jurnal
    banned_domains_or_paths = [
        "github.com", "flutter.dev", "facebook.com", "linkedin.com", "youtube.com",
        "twitter.com", "instagram.com", "wordpress.com", "blogspot.com", "play.google.com",
        "apps.apple.com", "shopee", "tokopedia", "support.google.com", "docs.flutter.dev",
        "developer.android.com", "creativecommons.org",
        "citation", "biblio", "metadata", "opcit", "indexing", "statistics", "rss", "atom",
        "contact", "faq", "contact-us", "about-us", "help", "privacy-policy", "terms-of-service",
        "conference", "agenda", "announcement", "registration", "site-map",
    ]

    allowed_journal_indicators = [
        "journal", "jurnal", "ejournal", "e-journal", "ojs", "scholar", "sciencedirect",
        "academic", "publication", "research", "article", "issue", "vol", "volume"
    ]

    if any(k in title_lower for k in stop_keywords):
        return False
    if any(k in link_lower for k in stop_keywords):
        return False
    if any(bad in link_lower for bad in banned_domains_or_paths):
        return False
    if not any(good in link_lower for good in allowed_journal_indicators):
        return False

    return True

# Banned domain agar tidak dikunjungi
def get_banned_domains():
    return [
        "github.com", "flutter.dev", "facebook.com", "linkedin.com", "youtube.com",
        "twitter.com", "instagram.com", "wordpress.com", "blogspot.com", "play.google.com",
        "apps.apple.com", "shopee", "tokopedia", "support.google.com", "docs.flutter.dev",
        "developer.android.com", "creativecommons.org",
        "citation", "biblio", "metadata", "opcit", "indexing", "statistics", "rss", "atom",
        "contact", "faq", "contact-us", "about-us", "help", "privacy-policy", "terms-of-service",
        "conference", "agenda", "announcement", "registration", "site-map",
    ]

irrelevant_anchor_keywords = [
    "login", "log in", "logout", "register", "sign in", "sign up", "submissions",
    "home", "editorial team", "publication ethics", "reviewer", "author guidelines",
    "online submission", "for librarians", "for readers", "user", "editor", "peer review", "review process",
    "site footer", "statcounter", "copyright", "journal template", "download", "peer review process",
    "open access", "open access statement", "license", "privacy", "terms", "archive", "contact", "faq",
    "about", "search", "cite", "skip", "Skip to site footer", "Skip to main navigation menu", 
    "formulir", "pdf", "focus and scope", "for authors", "http", "https", "disini", "plagiarism policy",
    "history", "current", "issn", "e-", "p-", "p-issn", "review guidelines", "publication fees",
    "view my stats", "publication frequency", "proses peer-review", "pedoman menulis", "akses terbuka",
    "undangan mitra bestar", "hak cipta dan lisensi", "author fee", "aim & scope", "publication fee", "in english",
    "out location", "view mystat", "screening", "open journal systems", "bahasa indonesia", "authors index", "indexing",
    "make", "sertifikat", "retraction index", "focus & scope", "scopus citation", "see visitor", "fast track review index", 
    "visitor statistics", "letter of acceptance", "http", "upgrade", "reports", "edit your profile", "exit link activity",
    "heatmaps", "traffic sources", "keyword activity", "page view activity", "visitor paths"
]

def is_relevant_anchor(anchor_text: str) -> bool:
    text = anchor_text.strip().lower()
    return not any(irrelevant in text for irrelevant in irrelevant_anchor_keywords)


# ===== Shark Search ====
def shark_search_crawler(query, max_depth=5, size=50, time_limit=900, width=20):
    frontier = []
    heap_counter = 0
    visited = set()
    relevant_results = []
    query_vec = embed_text(query)

    total_documents_visited = 0
    total_documents_evaluated = 0
    start_time = time.time()

    banned_domains = get_banned_domains()

    for seed_url in read_seed_urls():
        if any(banned in seed_url.lower() for banned in banned_domains):
            print(f"[SKIP] Seed URL melewati banned domain: {seed_url}")
            continue
        
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

    # while frontier and len(relevant_results) < MAX_RESULTS:
    while frontier:
        if total_documents_visited >= size:
            print("[STOP] Mencapai batas ukuran (size).")
            break
        if (time.time() - start_time) > time_limit:
            print("[STOP] Mencapai batas waktu (time_limit).")
            break

        _, _, node = heapq.heappop(frontier)
        url = node['url']
        inherited_score = node['inherited_score']
        depth = node['depth']

        if any(banned in url.lower() for banned in banned_domains):
            print(f"[SKIP] URL termasuk dalam banned domain: {url}")
            continue

        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        print(f"\n[DEPTH {depth}] Mengunjungi: {url}")
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            total_documents_visited += 1

            save_raw_html(soup, len(visited))

        except Exception as e:
            print(f"[ERROR] Gagal mengakses {url}: {e}")
            continue

        links = soup.find_all('a')
        print(f"[INFO] Ditemukan {len(links)} link di halaman ini.")

        scored_links = []
        for a in links:
            title = a.get_text(strip=True)
            anchor_href = a.get('href')
            anchor_url = urljoin(url, anchor_href) if anchor_href else None

            if not title:  # ✅ skip anchor dengan title kosong
                continue

            # ✅ Skip jika title hanya angka atau ID mirip ISSN
            if re.fullmatch(r"[0-9\-]+", title):
                continue
            
            if re.fullmatch(r'[\d\s\-()]+', title):
                continue

            # 3. Deteksi pola auto-template seperti CMS atau plugin codes
            if re.search(r'[#%{}<>]|plugin|label|block|##', title):
                continue
            
            if not anchor_url or not is_valid_url(anchor_url):
                continue

            if not is_relevant_anchor(title):
                print(f"[SKIP] Anchor dibuang karena tidak relevan: '{title}'")
                continue

            if any(banned in anchor_url.lower() for banned in banned_domains):
                print(f"[SKIP] Link anak diblokir karena termasuk banned domain: {anchor_url}")
                continue
                
            # Step by step by Shark Search Algorithm
            try:
                # Siapkan soup
                current_node_title = soup.title.string.strip() if soup.title and soup.title.string else ""
                current_node_relevance = predict_relevance(current_node_title)
                current_node_sim = cosine_sim(query_vec, embed_text(current_node_title)) if current_node_relevance > 0 else 0

                # Step 1 : Inherited Score
                if current_node_relevance > 0:
                    child_inherited_score = delta * current_node_sim
                    # print(f"[INHERITED] Relevan. Score = δ({delta}) × cosine_sim({current_node_sim:.4f}) = {child_inherited_score:.4f}")
                else:
                    child_inherited_score = delta * inherited_score
                    # print(f"[INHERITED] Tidak relevan. Score = δ({delta}) × parent_score({inherited_score:.4f}) = {child_inherited_score:.4f}")

                # Step 2 : Extract Anchor text & Context
                anchor_vec = embed_text(title)
                context_text = soup.get_text()[:500]
                anchor_context_vec = embed_text(context_text)

                # Step 3 : Anchor Score
                anchor_score = cosine_sim(query_vec, anchor_vec)

                # Step 4 : Anchor Context Score 
                if anchor_score > 0:
                    anchor_context_score = 1.0
                else:    
                    anchor_context_score = cosine_sim(query_vec, anchor_context_vec)
                
                # Step 5 : Neighbourhood Score
                neighborhood_score = beta * anchor_score + (1 - beta) * anchor_context_score

                # Step 6 : Potential Score
                potential_score = gamma * child_inherited_score + (1 - gamma) * neighborhood_score

                scored_links.append((potential_score, title, anchor_url, anchor_score, anchor_context_score, child_inherited_score, neighborhood_score))

            except Exception as e:
                print(f"[ERROR] Gagal proses link: {e}")
                continue

        # Batasi link anak berdasarkan width (ambil top-N berdasarkan skor disini menggunakan top 10 terbaik sesuai width = 10)
        scored_links.sort(reverse=True, key=lambda x: x[0])
        for i, (potential_score, title, anchor_url, anchor_score, context_score, child_inherited_score, neighborhood_score) in enumerate(scored_links[:width]):
            total_documents_evaluated += 1

            is_relevant = predict_relevance(title)
            valid_journal = is_valid_journal(title, anchor_url)
            print(f"[CHECK] '{title}' | Anchor Score: {anchor_score:.3f} | Relevan: {is_relevant} | Valid Journal: {valid_journal}")

            hasil = {
                'id': total_documents_evaluated,
                'title': title,
                'link': anchor_url,
                'anchor_score': float(f"{anchor_score:.4f}"),
                # 'context_score': float(f"{context_score:.4f}"),
                # 'inherited_score': float(f"{child_inherited_score:.4f}"),
                # 'decayed_parent_score': float(f"{child_inherited_score:.4f}"),
                # 'neighborhood_score': float(f"{neighborhood_score:.4f}"),
                'potential_score': float(f"{potential_score:.4f}"),
                'relevance': int(is_relevant)
            }
            
            # Tambahkan dan simpan ke results jika relevan dan terdeteksi sebagai jurnal
            if is_relevant == 1 and valid_journal:
                relevant_results.append(hasil) # Masuk ke json log 
                print(f"[RELEVAN] +1 → {len(relevant_results)} total")

            # Kondisi mengunjungi depth > 0 menggunakan acuan parameter threshold_score
            if potential_score >= threshold_score and depth < max_depth:
                heapq.heappush(frontier, (-potential_score, heap_counter, {
                    'url': anchor_url,
                    'depth': depth + 1,
                    'inherited_score': child_inherited_score
                }))
                heap_counter += 1

    harvest_rate = len(relevant_results) / total_documents_evaluated if total_documents_evaluated > 0 else 0.0
    print(f"\n[INFO] Total halaman web dikunjungi: {total_documents_visited}")
    print(f"[INFO] Total dokumen dievaluasi: {total_documents_evaluated}")
    print(f"[INFO] Total dokumen relevan: {len(relevant_results)}")
    print(f"[INFO] Harvest Rate (HR): {harvest_rate:.4f}")
    print(f"[INFO] Waktu berjalan: {int(time.time() - start_time)} detik")

    # save_results_to_json(relevant_results, 'results_pengujian_1_100.json')

    return relevant_results

# ===== BFS Crawler =====
def crawl_page(url, visited_urls, urls_to_visit, relevant_docs, irrelevant_docs):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        title_tag = soup.find('title')
        title_text = title_tag.get_text().strip() if title_tag else ''
        print(f"  Judul: {title_text}")

        if is_valid_journal(title_text, url):
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
        else:
            print("  ⛔ Bukan halaman jurnal")
            irrelevant_docs.append(url)

        for link in soup.find_all('a', href=True):
            absolute_url = urljoin(url, link['href'])
            parsed = urlparse(absolute_url)
            if parsed.scheme in ['http', 'https']:
                if absolute_url not in visited_urls and absolute_url not in urls_to_visit.queue and urls_to_visit.qsize() < 500:
                    urls_to_visit.put(absolute_url)

    except requests.exceptions.RequestException as e:
        print(f"  ⚠️ Gagal mengambil URL: {url} - {e}")

    return visited_urls, urls_to_visit, relevant_docs, irrelevant_docs

def bfs_crawler(seed_urls, max_page):
    seed_urls = read_seed_urls()
    visited_urls = set()
    urls_to_visit = queue.Queue()

    for url in seed_urls:
        urls_to_visit.put(url)

    relevant_docs = []
    irrelevant_docs = []
    total_visited = 0

    while not urls_to_visit.empty() and total_visited < max_page:
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

# if __name__ == "__main__":
#     # max_pages = 50
#     # query = "Pengembangan Aplikasi Berbasis Android Menggunakan Flutter" 
#     query = "Kecerdasan Buatan Masa Depan Akuntansi" 
#     results = shark_search_crawler(query, max_depth=3, size=100, time_limit=400, width=10)
#     print(f"\n[SELESAI] Total dokumen relevan ditemukan: {len(results)}")


if __name__ == '__main__':
    mode = input("Pilih mode crawler (bfs/shark): ").strip().lower()
    if mode == 'bfs':
        seed_urls = read_seed_urls()
        bfs_crawler(seed_urls, max_page=100)
    elif mode == 'shark':
        query = input("Masukkan query pencarian: ")
        shark_search_crawler(query, max_depth=5, size=50, time_limit=900, width=20)
    else:
        print("Mode tidak dikenali.")