import requests

def get_google_scholar_seed_urls(query, api_key, limit = 10):
  search_url = "https://serpapi.com/search"
  params = {
    'engine': 'google_scholar',
    'q': query,
    'api_key': api_key
  }

  keyword_filters = ["jurnal", "journal", "ejournal", "e-journals"]

  try:
    response = requests.get(search_url, params=params)
    response.raise_for_status()
    data = response.json()

    seed_urls = []
    for result in data.get("organic_results", []):
      link = result.get("link", "")
      
      if link.endswith(".pdf") or "/pdf" in link or "download" in link:
        continue
      
      if any(keyword in link.lower() for keyword in keyword_filters):
        seed_urls.append(link)

      if len(seed_urls) >= limit:
        break
    
    return seed_urls
  
  except requests.exceptions.RequestException as e:
    print(f"error bro: {e}")
    return []

api_key = "6449b5104e44f9797dadb2294dfb3dd42a08ba77d533faad4e6be932066d67e1"
query = "Pengembangan Aplikasi Berbasis Android Menggunakan Flutter"
seed_urls = get_google_scholar_seed_urls(query, api_key)

for url in seed_urls:
  print(url)