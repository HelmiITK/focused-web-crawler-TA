import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
import re

# Load KNN model
with open('knn_model_k4.pkl', 'rb') as file:
    knn_model = joblib.load(file)

# Load IndoBERT
tokenizer = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model = AutoModel.from_pretrained('indobenchmark/indobert-base-p1')

def clean_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()


def bert_embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True,  max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # Ambil [CLS] token
    return embeddings.numpy()

def compute_similarity(query, text):
    query_vec = bert_embed(query)
    text_vec = bert_embed(text)
    score = cosine_similarity(query_vec, text_vec)[0][0]
    return score

def is_relevant(title_text):
    # Preprocessing text
    title_clean = clean_text(title_text)
    title_vec = bert_embed(title_clean)
    return knn_model.predict(title_vec)[0] == 1
