import sqlite3

def init_db():
    conn = sqlite3.connect('plant_data.db')
    c = conn.cursor()

    # 1. Tabel untuk menyimpan informasi penyakit & solusi (Knowledge Base)
    c.execute('''
        CREATE TABLE IF NOT EXISTS diseases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT UNIQUE NOT NULL,
            description TEXT,
            treatment TEXT,
            prevention TEXT
        )
    ''')

    # 2. Tabel untuk menyimpan riwayat scan user (History Log)
    c.execute('''
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 3. Masukkan Data Awal (Seeding)
    # Data ini bisa Anda tambah/edit tanpa mengubah kodingan app.py
    diseases_data = [
        ("Potato___Early_blight", 
         "Penyakit Bercak Kering (Early Blight) disebabkan oleh jamur *Alternaria solani*.",
         "1. Pangkas daun yang terinfeksi.\n2. Gunakan fungisida berbahan aktif Klorotalonil.",
         "Jaga kelembaban tanah dan beri jarak tanam yang cukup."),
        
        ("Potato___Late_blight", 
         "Busuk Daun (Late Blight) adalah penyakit serius yang disebabkan oleh *Phytophthora infestans*.",
         "1. Musnahkan tanaman yang parah.\n2. Gunakan fungisida sistemik (Metalaksil).",
         "Hindari penyiraman sore hari dan gunakan bibit unggul tahan penyakit."),
        
        ("Potato___healthy", 
         "Tanaman Kentang Anda dalam kondisi prima!",
         "Tidak ada penanganan khusus yang diperlukan.",
         "Lanjutkan pemupukan rutin dan monitoring hama."),
         
         # Tambahkan kelas lain jika Anda punya (misal Tomat dll)
    ]

    c.executemany('''
        INSERT OR IGNORE INTO diseases (class_name, description, treatment, prevention)
        VALUES (?, ?, ?, ?)
    ''', diseases_data)

    conn.commit()
    conn.close()
    print("Database plant_data.db berhasil dibuat dan diisi data awal!")

if __name__ == '__main__':
    init_db()