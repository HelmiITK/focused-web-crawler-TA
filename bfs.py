peta = {
  'A': set (['B', 'H']),
  "B": set (['A', 'H', 'C']),
  "C": set (['B', 'D', 'E']),
  "D": set (['C', 'E', 'F', 'G', 'H']),
  "E": set (['C', 'D']),
  "F": set (['D', 'G']),
  "G": set (['H', 'D', 'F']),
  "H": set (['A', 'B', 'D', 'G']),
}

def bfs_lintasan_terpendek(peta, mulai, tujuan):
  explore = []
  queue = [[mulai]]

  if mulai == tujuan :
    return "Awal adalah tujuan"
  
  while queue : 
    jalur = queue.pop(0)
    node = jalur[-1]

    if node not in explore :
      neighbours = peta[node]

      for neighbour in neighbours :
        jalur_baru = list(jalur)
        jalur_baru.append(neighbour)
        queue.append(jalur_baru)

        if neighbour == tujuan :
          return jalur_baru
        
      explore.append(node)

  return "Mohon maaf node yang di pilih tidak ada di peta"

mulai = input("Masukka Node Awal: ")
tujuan = input("Masukka Node Akhir: ")

print(bfs_lintasan_terpendek(peta, mulai, tujuan))
