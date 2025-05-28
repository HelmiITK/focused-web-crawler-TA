import requests

def get_google_scholar_seed_urls(query, api_key, limit=20, max_pages=5):
    search_url = "https://serpapi.com/search"
    keyword_filters = ["jurnal", "journal", "ejournal", "e-journals"]

    seed_urls = []
    try:
        for page in range(max_pages):
            start = page * 10
            params = {
                'engine': 'google_scholar',
                'q': query,
                'api_key': api_key,
                'start': start
            }

            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()

            for result in data.get("organic_results", []):
                link = result.get("link", "")

                if link.endswith(".pdf") or "/pdf" in link or "download" in link:
                    continue

                if any(keyword in link.lower() for keyword in keyword_filters):
                    if link not in seed_urls:  # Hindari duplikasi
                        seed_urls.append(link)

                if len(seed_urls) >= limit:
                    return seed_urls

    except requests.exceptions.RequestException as e:
        print(f"error bro: {e}")

    return seed_urls

api_key = "6449b5104e44f9797dadb2294dfb3dd42a08ba77d533faad4e6be932066d67e1"
query = "Pengembangan Aplikasi Berbasis Android Menggunakan Flutter "
seed_urls = get_google_scholar_seed_urls(query, api_key)

for url in seed_urls:
    print(url)