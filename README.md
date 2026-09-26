# Eksperimen Klasifikasi Sentimen: Model Klasik vs LLM API (Gemini)

Eksperimen awal untuk membandingkan dua pendekatan klasifikasi sentimen ulasan pelanggan e-commerce (positif/negatif), sebagai bahan pertimbangan tim produk dalam memilih pendekatan untuk sistem produksi:

1. **Model klasik**: TF-IDF + Logistic Regression (Scikit-learn), dilatih dari dataset ulasan.
2. **LLM API**: Gemini dengan prompt zero-shot, tanpa proses training.

Notebook lengkap: [notebook/starter_notebook.ipynb](notebook/starter_notebook.ipynb)

## 1. Problem Statement

### Objective Use Case
Menentukan pendekatan yang sesuai untuk membangun fitur otomatis klasifikasi sentimen ulasan pelanggan (positif/negatif) pada halaman produk, dari perbandingan eksperimen menggunakan model klasik (Scikit-learn) dan LLM API (Gemini).

### Target/Label
Bentuk label 1/0
- 1 (Positif) : Ulasan bersifat positif (menyatakan kepuasan, pujian, atau rekomendasi) terhadap produk
- 0 (Negatif) : Ulasan bersifat negatif (menyatakan kekecewaan, keluhan, atau kritik) terhadap produk

### Batasan dan Asumsi
- Klasifikasi biner (positif/negatif); ulasan netral tidak digunakan
- Dataset yang digunakan bahasa Indonesia dengan jumlah 200 ulasan, tetapi hanya 40 ulasan yang unik (sisanya duplikat)
- Metrik penilaian yang dipakai yaitu accuracy, precision, recall, F1, dan latency. Biaya per prediksi dibahas secara kualitatif
- Label di dataset dianggap benar dan mewakili sentimen sebenarnya
- Kedua pendekatan model klasik maupun LLM API dievaluasi pada test set yang sama supaya perbandingannya adil
- Pembagian train dan test set sebesar 80/20 (160 train, 40 test)
- Untuk awal percobaan Gemini dipakai secara zero-shot tanpa fine-tuning, dengan temperature 0 agar output konsisten.
- Karena dataset kecil, hasil eksperimen belum tentu mewakili performa pada data produksi.

## 2. Struktur Repository

```
├── data/
│   └── customer_reviews_sentiment.csv   # dataset (200 baris, kolom review_text & sentiment)
├── documentation/
│   ├── comparison-approach.png          # tangkapan layar tabel perbandingan
│   ├── tabel_perbandingan.csv           # ringkasan metrik & latency kedua pendekatan
│   ├── hasil_prediksi_klasik.csv        # prediksi model klasik per ulasan
│   ├── hasil_prediksi_llm.csv           # prediksi & output mentah Gemini per ulasan
│   ├── klasik_metrik.csv / llm_metrik.csv
│   └── klasik_confusion_matrix.csv / llm_confusion_matrix.csv
├── notebook/
│   └── starter_notebook.ipynb           # eksperimen, evaluasi, analisis & rekomendasi
├── requirements.txt
└── README.md
```

## 3. Cara Menjalankan

**Google Colab**
1. Upload `notebook/starter_notebook.ipynb` dan `data/customer_reviews_sentiment.csv` (lewat panel Files).
2. Tambahkan API key Gemini di 🔑 **Secrets** dengan nama `GEMINI_API_KEY`, lalu aktifkan *Notebook access*.
3. Jalankan semua cell dari atas (Runtime → Run all).

**Lokal**
1. Install dependency:
   ```
   pip install -r requirements.txt
   ```
2. Simpan API key Gemini di environment variable `GEMINI_API_KEY`.
3. Buka `notebook/starter_notebook.ipynb` dan jalankan semua cell. Path dataset (`../data/...`) dan API key terdeteksi otomatis.

## 4. Setup Eksperimen

| | Model Klasik | LLM API |
|---|---|---|
| Model | TF-IDF + Logistic Regression | `gemini-3.1-flash-lite` |
| Training | Dilatih dengan data train | Tanpa training (zero-shot) |
| Parameter | Default Scikit-learn | Temperature 0, output 1 kata (`positif`/`negatif`), retry otomatis saat error 503 |
| Output | Label 0/1 | Teks, dinormalisasi menjadi 0/1 |

Kedua pendekatan dievaluasi pada test set yang sama (`train_test_split` dengan `test_size=0.2`, `random_state=42`, `stratify`).

## 5. Hasil Evaluasi

| Pendekatan | Accuracy | Precision | Recall | F1-Score | Latency/ulasan |
|---|---|---|---|---|---|
| Model Klasik (TF-IDF + Logistic Regression) | 1.0 | 1.0 | 1.0 | 1.0 | 0,06 ms |
| LLM API (gemini-3.1-flash-lite, zero-shot) | 1.0 | 1.0 | 1.0 | 1.0 | ±4.727 ms (±4,7 detik) |

Confusion matrix kedua pendekatan sama, dengan test set 40 ulasan (18 negatif, 22 positif) dan tidak ada salah prediksi:

| | Prediksi negatif | Prediksi positif |
|---|---|---|
| **Asli negatif** | 18 | 0 |
| **Asli positif** | 0 | 22 |

**Catatan dengan vs tanpa drop duplicate:** skenario awal memakai drop duplicate (32 train / 8 test). Hasil akhir di atas menggunakan data tanpa drop duplicate (160 train / 40 test). Skor kedua skenario tetap sama (1.0), karena datanya sedikit dan ulasannya relatif mirip. Namun tanpa drop duplicate, 25 dari 26 ulasan unik di test set juga ada di data train, sehingga skor model klasik perlu dibaca dengan hati-hati. Detailnya ada di Section 7.3 notebook.

## 6. Rekomendasi

Dari sisi akurasi, eksperimen ini **belum bisa menentukan pendekatan yang lebih baik**, karena keduanya sama-sama mendapat skor 1.0 pada dataset yang kecil dan mudah. Perbedaan yang jelas ada pada kecepatan, biaya, dan effort:

- **Model klasik** jauh lebih cepat (milidetik vs detik) dan hampir tanpa biaya, tetapi membutuhkan dataset berlabel yang bagus.
- **LLM API** tidak membutuhkan dataset dan training, tetapi lebih lambat, ada biaya per panggilan, dan bergantung pada ketersediaan server Google (selama eksperimen sempat terjadi error 503 dan beberapa versi model sudah tidak tersedia).

**Rekomendasi:** gunakan **model klasik untuk produksi jangka panjang**, dengan syarat tim menyiapkan dataset yang lebih besar dan beragam. Gunakan **LLM API** jika fitur perlu cepat jadi tanpa menyiapkan dataset. Kombinasi keduanya juga bisa dipertimbangkan: Gemini membantu memberi label awal (dicek manusia), lalu hasilnya dipakai untuk melatih model klasik.

Analisis trade-off, limitation, dan rekomendasi lengkap ada di Section 7 dan 8 [notebook](notebook/starter_notebook.ipynb).
