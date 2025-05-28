import joblib
from transformers import AutoTokenizer, AutoModel
import torch

# 1. Load model IndoBERT
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")

# 2. Load model KNN
knn = joblib.load('save_model_knn/knn_model_k4.pkl') # Paramter K = 4 Terbaik 

# 3. Fungsi untuk mendapatkan embedding IndoBERT
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:,0,:].squeeze().numpy()
    return cls_embedding

# 4. Fungsi untuk prediksi teks baru
def predict_text(text):
    embedding = get_embedding(text)
    prediction = knn.predict(embedding.reshape(1, -1))[0]
    return prediction

# 5. Contoh penggunaan
if __name__ == "__main__":
    # teks_baru = "analisis keamanan pada teknologi blockchain"
    teks_baru = "Home Adalah rumah"
    hasil = predict_text(teks_baru)
    print(f"Hasil prediksi untuk teks: '{teks_baru}' adalah {hasil}")
