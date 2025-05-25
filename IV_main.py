import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from III_focused_crawler import focused_crawler_shark_search

def load_seed_urls(path="seed_urls_dengan_judul.txt"):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

if __name__ == "__main__":
    seeds = load_seed_urls()
    if seeds:
        result = focused_crawler_shark_search(seed_urls=seeds)
        print("\nHasil crawling:")
        for url, relevan in result:
            print(f"{url} | relevan: {relevan}")
    else:
        print("Tidak ada URL awal untuk crawling.")
