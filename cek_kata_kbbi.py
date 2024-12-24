# Membaca file yang berisi kumpulan kata bahasa Indonesia (KBBI)
file_path = "indonesian-words.txt"

# Kata yang ingin dicek dan ditambahkan jika tidak ditemukan
# word_to_check = "dengan"
word_to_check = input("Masukkan Kata:").strip().lower()

try:
    # Membuka dan membaca file
    with open(file_path, "r", encoding="utf-8") as file:
        # Membaca semua kata dan menghapus spasi putih di sekitar
        words = [line.strip() for line in file.readlines()]

    # Mengecek apakah kata ada dalam file
    if word_to_check in words:
        print(f"Kata '{word_to_check}' ditemukan di file.")
    else:
        print(f"Kata '{word_to_check}' tidak ditemukan di file.")
        
        # Meminta konfirmasi dari pengguna
        user_input = input("Apakah Anda ingin menambahkan kata ini ke file? (y/n): ").strip().lower()
        if user_input == 'y':
            # Menambahkan kata ke daftar
            words.append(word_to_check)
            
            # Mengurutkan daftar kata sesuai alfabet
            words = sorted(words)
            
            # Menulis kembali daftar kata ke file
            with open(file_path, "w", encoding="utf-8") as file:
                for word in words:
                    file.write(word + "\n")
            
            print(f"Kata '{word_to_check}' berhasil ditambahkan ke file.")
        else:
            print("Kata tidak ditambahkan.")
except FileNotFoundError:
    print(f"File '{file_path}' tidak ditemukan. Membuat file baru...")
    
    # # Membuat file baru dan menambahkan kata
    # with open(file_path, "w", encoding="utf-8") as file:
    #     file.write(word_to_check + "\n")
    # print(f"File baru dibuat dengan kata '{word_to_check}'.")
