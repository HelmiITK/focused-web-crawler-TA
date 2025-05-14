import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# 1. Load dataset yang sudah dilabeli
df = pd.read_csv('pre_processing/label/combined_labeled.csv')

# 2. Load IndoBERT model dan tokenizer
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")

# Fungsi untuk mengubah teks menjadi embedding IndoBERT
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # Ambil CLS token output (biasanya [0,0,:])
    cls_embedding = outputs.last_hidden_state[:,0,:].squeeze().numpy()
    return cls_embedding

# 3. Convert semua Judul menjadi embedding IndoBERT
embeddings = []
for text in df['Judul']:
    embedding = get_embedding(text)
    embeddings.append(embedding)

# Simpan sebagai .npy (efisien)
np.save('pre_processing/embedding/combined_embedding.npy', np.array(embeddings))

# Simpan label
df['Label'].to_csv('pre_processing/embedding/combined_label.csv', index=False)

# import numpy as np

# # Load file .npy
# embeddings = np.load('pre_processing/embedding/combined_embedding.npy')

# # Cek bentuk data
# print("Shape of embeddings:", embeddings.shape)  # Misalnya: (1930, 768)

# # Lihat 5 baris pertama
# # print("First 5 embeddings:")
# # print(embeddings[:5])
# print(embeddings[1])
