import matplotlib.pyplot as plt

# Data yang diperbarui
pages_visited = [25, 50, 75, 100]
harvest_rate_bfs = [0.56, 0.52, 0.44, 0.50]
harvest_rate_shark = [0.9289, 0.927, 0.9244, 0.9187]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(pages_visited, harvest_rate_bfs, marker='o', label='BFS', color='blue')
plt.plot(pages_visited, harvest_rate_shark, marker='s', label='Shark Search', color='green')

# Labeling
plt.title('Harvest Rate: BFS vs Shark Search')
plt.xlabel('Halaman Yang Dicrawl (size)')
plt.ylabel('Harvest Rate')
plt.ylim(0.4, 1.0)
plt.grid(True)
plt.legend()
plt.xticks(pages_visited)

# Simpan grafik sebagai PNG
output_path = 'harvest_rate_perbandingan_SS_BFS.png'
plt.tight_layout()
plt.savefig(output_path)
plt.show()

output_path
