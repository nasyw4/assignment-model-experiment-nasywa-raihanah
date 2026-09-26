## 1. Objective Use Case
Menentukan pendekatan yang sesuai untuk membangun fitur otomatis klasifikasi sentimen ulasan pelanggan (positif/negatif) pada halaman produk, dari perbandingan eksperimen menggunakan model klasik (Scikit-learn) dan LLM API (Gemini).

## 2. Target/Label
Bentuk label 1/0
- 1 (Positif) : Ulasan bersifat positif (menyatakan kepuasan, pujian, atau rekomendasi) terhadap produk
- 0 (Negatif) : Ulasan bersifat negatif (menyatakan kekecewaan, keluhan, atau kritik) terhadap produk

## 3. Batasan dan Asumsi
- Klasifikasi biner (positif/negatif); ulasan netral tidak digunakan
- Dataset yang digunakan bahasa Indonesia dengan jumlah 200 ulasan
- Metrik penilaian yang dipakai yaitu accuracy, precision, recall, F1, latency, dan biaya per prediksi
- Label di dataset dianggap benar dan mewakili sentimen sebenarnya
- Kedua pendekatan model klasik maupun LLM API dievaluasi pada test set yang sama supaya perbandingannya adil
- Pembagian train dan test set sebesar 80/20 (160 train, 40 test)
- Untuk awal percobaan Gemini dipakai secara zero-shot tanpa fine-tuning, dengan temperature 0 agar output konsisten.
- Karena dataset kecil (200 ulasan), hasil eksperimen belum tentu mewakili performa pada data produksi.
