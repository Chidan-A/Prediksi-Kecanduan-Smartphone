import joblib
import pandas as pd

# Muat model dari file
try:
    nb = joblib.load('smartphone_addiction_model.pkl')
    print("Model berhasil dimuat.")
except FileNotFoundError:
    print("Error: File 'smartphone_addiction_model.pkl' tidak ditemukan. Pastikan file model berada di direktori yang sama.")
    exit()
except Exception as e:
    print(f"Terjadi kesalahan saat memuat model: {e}")
    exit()

# --- Menggunakan Model untuk Prediksi Data Baru ---
print("\nMasukkan data baru untuk memprediksi kecanduan smartphone:")

try:
    Age = float(input("Masukkan umur anda: "))
    Academic_Performance = float(input("Berapakah nilai akademik anda?: "))
    Social_Interactions = float(input("Berapa jam interaksi sosial anda?: "))

    # Buat DataFrame dari input baru
    new_data_df = pd.DataFrame(
        [[Age, Academic_Performance, Social_Interactions]],
        columns=['Age', 'Academic_Performance', 'Social_Interactions']
    )

    # Lakukan prediksi
    predicted_code = nb.predict(new_data_df)[0]  # hasilnya 0, 1, atau 2

    # Konversi hasil prediksi ke label asli
    label_mapping = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
    predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

    print(f"\nUntuk data tersebut:")
    print(f"Prediksi kecanduan smartphone adalah: {predicted_label}")

except ValueError:
    print("Input tidak valid. Harap masukkan angka.")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")