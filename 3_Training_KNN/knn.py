import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load data
X = np.load('D:\\Semester 8\\TA\\focused-web-crawler\\2_Preprocessing\\embedding\\combined_embedding.npy')
y = pd.read_csv('D:\\Semester 8\TA\\focused-web-crawler\\2_Preprocessing\\embedding\\combined_label.csv').values.ravel()

# Split dan train
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=4, metric='cosine')
# knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# joblib.dump(knn, 'save_model_knn/knn_model_k4.pkl')  # Simpan model knn
# print("Model KNN berhasil disimpan ke 'save_model_knn/knn_model_k4.pkl'")

# Evaluasi
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Hitung metrik
TN, FP, FN, TP = cm.ravel()
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

# Cetak hasilnya
print(f"Classification Report: \n {classification_report(y_test, y_pred)}")
print(f"K : {knn.n_neighbors}" )
print(f"Confusion Matrix : \n {cm}")
print(f"Accuracy  : {accuracy_score(y_test, y_pred)}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1_score:.4f}")


sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Unrelevan', 'Relevan'], yticklabels=['Unrelevan', 'Relevan'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix') 
# plt.savefig('metrix_evaluation/confusion_matrix_k20.png')
plt.show()


#========================================
# Grafik K 1 - n

# import numpy as np
# import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import os

# # Load embedding dan label
# embeddings = np.load('pre_processing/embedding/combined_embedding.npy')
# df = pd.read_csv('pre_processing/label/combined_labeled.csv')
# labels = df['Label'].values

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     embeddings, labels, test_size=0.2, random_state=42, stratify=labels
# )

# # Siapkan range nilai K
# k_values = range(1, 21)
# accuracy_scores = []
# f1_scores = []

# # Uji semua K
# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
#     knn.fit(X_train, y_train)
#     y_pred = knn.predict(X_test)

#     acc = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     accuracy_scores.append(acc)
#     f1_scores.append(f1)

# # Plot grafik
# plt.figure(figsize=(10, 6))
# plt.plot(k_values, accuracy_scores, label='Accuracy', marker='o')
# plt.plot(k_values, f1_scores, label='F1 Score', marker='s')
# plt.title('Accuracy dan F1 Score terhadap Nilai K')
# plt.xlabel('Nilai K')
# plt.ylabel('Skor')
# plt.xticks(k_values)
# plt.grid(True)
# plt.legend()

# # Simpan grafik
# os.makedirs('metrix_evaluation', exist_ok=True)
# plt.savefig('metrix_evaluation/knn_k_comparison_1-20.png')
# plt.show()

#========================================
#PErbandingan Cosine dan Euclidean

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

# # Load data
# X = np.load('pre_processing/embedding/combined_embedding.npy')
# y = pd.read_csv('pre_processing/embedding/combined_label.csv').values.ravel()

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# # Simpan akurasi
# k_values = list(range(1, 11))
# accuracy_euclidean = []
# accuracy_cosine = []

# for k in k_values:
#     knn_euc = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
#     knn_euc.fit(X_train, y_train)
#     accuracy_euclidean.append(accuracy_score(y_test, knn_euc.predict(X_test)))

#     knn_cos = KNeighborsClassifier(n_neighbors=k, metric='cosine')
#     knn_cos.fit(X_train, y_train)
#     accuracy_cosine.append(accuracy_score(y_test, knn_cos.predict(X_test)))

# # Visualisasi
# plt.figure(figsize=(10, 6))
# plt.plot(k_values, accuracy_euclidean, marker='o', label='Euclidean Distance')
# plt.plot(k_values, accuracy_cosine, marker='o', label='Cosine Similarity')
# plt.title('Perbandingan Akurasi KNN dengan Euclidean vs Cosine')
# plt.xlabel('Nilai K')
# plt.ylabel('Akurasi')
# plt.xticks(k_values)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()