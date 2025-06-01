import matplotlib.pyplot as plt

#====================DELTA==============================

# # Data eksperimen berdasarkan variasi parameter delta
# delta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# harvest_rates = [0.9204, 0.9204, 0.9194, 0.9289, 0.9218, 0.9189, 0.9212, 0.9212, 0.9171, 0.9118]
# evaluated_docs = [427, 427, 422, 436, 422, 419, 419, 419, 398, 397]
# relevant_docs = [393, 393, 388, 405, 389, 385, 386, 386, 365, 362]
# runtime_seconds = [173, 175, 149, 164, 152, 145, 142, 149, 135, 157]

# # Membuat plot
# plt.figure(figsize=(14, 8))

# # Harvest rate
# plt.subplot(2, 2, 1)
# plt.plot(delta_values, harvest_rates, marker='o', color='green')
# plt.title('Harvest Rate vs Delta')
# plt.xlabel('Delta')
# plt.ylabel('Harvest Rate')
# plt.grid(True)

# # Dokumen dievaluasi
# plt.subplot(2, 2, 2)
# plt.plot(delta_values, evaluated_docs, marker='s', color='blue')
# plt.title('Dokumen Dievaluasi vs Delta')
# plt.xlabel('Delta')
# plt.ylabel('Jumlah Dokumen Dievaluasi')
# plt.grid(True)

# # Dokumen relevan
# plt.subplot(2, 2, 3)
# plt.plot(delta_values, relevant_docs, marker='^', color='orange')
# plt.title('Dokumen Relevan vs Delta')
# plt.xlabel('Delta')
# plt.ylabel('Jumlah Dokumen Relevan')
# plt.grid(True)

# # Waktu berjalan
# plt.subplot(2, 2, 4)
# plt.plot(delta_values, runtime_seconds, marker='x', color='red')
# plt.title('Waktu Berjalan (detik) vs Delta')
# plt.xlabel('Delta')
# plt.ylabel('Waktu (detik)')
# plt.grid(True)

# # Simpan grafik sebagai PNG
# output_path = 'evaluasi_parameter_shark_search/delta.png'
# plt.savefig(output_path)

# output_path

# plt.tight_layout()
# plt.show()

#====================BETA==============================

# # Data eksperimen berdasarkan variasi parameter beta
# beta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# harvest_rates = [0.8855, 0.9118, 0.9189, 0.9218, 0.9289, 0.9194, 0.9194, 0.9194, 0.9204, 0.9204]
# evaluated_docs = [428, 397, 419, 422, 436, 422, 422, 422, 427, 427]
# relevant_docs = [379, 362, 385, 389, 405, 388, 388, 388, 393, 393]
# runtime_seconds = [171, 138, 140, 139, 142, 139, 171, 165, 178, 173]

# # Membuat plot
# plt.figure(figsize=(14, 8))

# # Harvest rate
# plt.subplot(2, 2, 1)
# plt.plot(beta_values, harvest_rates, marker='o', color='green')
# plt.title('Harvest Rate vs Beta')
# plt.xlabel('Beta')
# plt.ylabel('Harvest Rate')
# plt.grid(True)

# # Dokumen dievaluasi
# plt.subplot(2, 2, 2)
# plt.plot(beta_values, evaluated_docs, marker='s', color='blue')
# plt.title('Dokumen Dievaluasi vs Beta')
# plt.xlabel('Beta')
# plt.ylabel('Jumlah Dokumen Dievaluasi')
# plt.grid(True)

# # Dokumen relevan
# plt.subplot(2, 2, 3)
# plt.plot(beta_values, relevant_docs, marker='^', color='orange')
# plt.title('Dokumen Relevan vs Beta')
# plt.xlabel('Beta')
# plt.ylabel('Jumlah Dokumen Relevan')
# plt.grid(True)

# # Waktu berjalan
# plt.subplot(2, 2, 4)
# plt.plot(beta_values, runtime_seconds, marker='x', color='red')
# plt.title('Waktu Berjalan (detik) vs Beta')
# plt.xlabel('Beta')
# plt.ylabel('Waktu (detik)')
# plt.grid(True)

# # Simpan grafik sebagai PNG
# output_path = 'evaluasi_parameter_shark_search/beta.png'
# plt.savefig(output_path)

# output_path

# plt.tight_layout()
# plt.show()

#====================GAMMA==============================

import matplotlib.pyplot as plt

# Data gamma
gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
evaluated_docs = [427, 427, 427, 378, 436, 419, 388, 123, 38, 38]
relevant_docs = [393, 393, 393, 306, 405, 385, 355, 115, 37, 37]
harvest_rates = [0.9204, 0.9204, 0.9204, 0.8095, 0.9289, 0.9189, 0.9149, 0.935, 0.9737, 0.9737]
runtime_seconds = [168, 183, 165, 193, 177, 173, 149, 42, 15, 13]
pages_visited = [25, 25, 25, 25, 25, 25, 25, 7, 2, 2]

# Plotting
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(gamma_values, harvest_rates, marker='o', color='green')
plt.title('Harvest Rate vs Gamma')
plt.xlabel('Gamma')
plt.ylabel('Harvest Rate')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(gamma_values, evaluated_docs, marker='o', label='Dievaluasi')
plt.plot(gamma_values, relevant_docs, marker='o', label='Relevan')
plt.title('Dokumen vs Gamma')
plt.xlabel('Gamma')
plt.ylabel('Jumlah Dokumen')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(gamma_values, runtime_seconds, marker='o', color='red')
plt.title('Waktu Berjalan vs Gamma')
plt.xlabel('Gamma')
plt.ylabel('Detik')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(gamma_values, pages_visited, marker='o', color='purple')
plt.title('Halaman Web Dikunjungi vs Gamma')
plt.xlabel('Gamma')
plt.ylabel('Jumlah Halaman')
plt.grid(True)

# Simpan grafik sebagai PNG
output_path = 'evaluasi_parameter_shark_search/gamma.png'
plt.savefig(output_path)

output_path

plt.tight_layout()
plt.show()
