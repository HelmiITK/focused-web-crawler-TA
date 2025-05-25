import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import time
import heapq
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from II_utils import compute_similarity, is_relevant

# Parameter Shark Search
WIDTH = 10
max_depth = 3
SIZE = 500
TIME_LIMIT = 600 
DELTA = 0.7
BETA = 0.6
GAMMA = 0.5
QUERY = "Machine Learning Untuk Pemula"

class Node:
    def __init__(self, url, depth, inherited_score=0):
        self.url = url
        self.depth = depth
        self.inherited_score = inherited_score
        self.potential_score = 0

    def __lt__(self, other):
        return self.potential_score > other.potential_score 

def focused_crawler_shark_search(query=QUERY, seed_urls=None, max_depth=max_depth, width=WIDTH, size=SIZE, time_limit=TIME_LIMIT):
    if not seed_urls:
        raise ValueError("Seed URLs tidak boleh kosong!")
    
    start_time = time.time()
    visited = set()
    frontier = []

    for url in seed_urls:
        node = Node(url, depth=max_depth)
        heapq.heappush(frontier, node)

    result = []
    processed_count = 0

    while frontier and processed_count < size and time.time() - start_time < time_limit:
        current_node = heapq.heappop(frontier)
        if current_node.url in visited:
            continue

        visited.add(current_node.url)
        processed_count += 1

        try:
            response = requests.get(current_node.url, timeout=20, verify=False)
            soup = BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.SSLError as ssl_err:
            print(f"SSL Error {current_node.url}: {ssl_err}")
            continue
        except requests.exceptions.ReadTimeout as timeout_err:
            print(f"Timeout {current_node.url}: {timeout_err}")
            continue
        except Exception as e:
            print(f"Error Bro {current_node.url}: {e}")
            continue
        
        title = soup.title.string if soup.title else ""
        rel = is_relevant(title)
        result.append((current_node.url, rel))

        relevance = rel

        if current_node.depth > 0:
            anchors = soup.find_all('a', href=True)
            for anchor in anchors[:width]:
                # child_url = anchor['href']
                # if not child_url.startswith('http'):
                #     continue
                child_url = urljoin(current_node.url, anchor['href']) 

                anchor_text = anchor.get_text(strip=True)
                context = anchor.find_parent().get_text(strip=True) if anchor.find_parent() else ""

                # Step 1: Inherited score
                if relevance:
                    title_text = soup.title.string if soup.title and soup.title.string else ""
                    inherited_score = DELTA * compute_similarity(query, title_text)
                else:
                    inherited_score = DELTA * current_node.inherited_score

                # Step 3: Anchor score
                anchor_score = compute_similarity(query, anchor_text)

                # Step 4: Anchor context score
                if anchor_score > 0:
                    anchor_context_score = 1
                else:
                    anchor_context_score = compute_similarity(query, context)

                # Step 5: Neighbourhood score
                neighbourhood_score = BETA * anchor_score + (1 - BETA) * anchor_context_score

                # Step 6: Potential score
                potential_score = GAMMA * inherited_score + (1 - GAMMA) * neighbourhood_score

                # Masukkan ke frontier
                next_depth = current_node.depth - 1 if current_node.depth > 0 else 0
                child_node = Node(child_url, next_depth, inherited_score)

                child_node.potential_score = potential_score

                if child_url not in visited:
                    heapq.heappush(frontier, child_node)

    return result