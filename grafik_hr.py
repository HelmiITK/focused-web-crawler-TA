import matplotlib.pyplot as plt

# Data
pages_visited = [25, 50, 100]
harvest_rate_bfs = [0.56, 0.52, 0.44]
harvest_rate_shark = [0.8432, 0.8404, 0.8182]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(pages_visited, harvest_rate_bfs, marker='o', label='BFS', color='blue')
plt.plot(pages_visited, harvest_rate_shark, marker='s', label='Shark Search', color='green')

# Labeling
plt.title('Perbandingan Harvest Rate: BFS vs Shark Search')
plt.xlabel('Halaman Yang Dicrawl')
plt.ylabel('Harvest Rate')
plt.ylim(0.4, 0.9)
plt.grid(True)
plt.legend()
plt.xticks(pages_visited)

# Simpan grafik sebagai PNG
output_path = 'harvest_rate_percobaan_1.png'
plt.savefig(output_path)

output_path

# Show plot
plt.tight_layout()
plt.show()
