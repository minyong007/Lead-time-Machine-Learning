# Prediksi Lead Time Barang Elektronik

## Deskripsi Proyek

Proyek ini bertujuan untuk membangun model machine learning sederhana yang dapat memprediksi lead time (waktu tunggu) pengadaan barang elektronik berdasarkan data riwayat order. Dengan mengetahui lead time secara lebih akurat, perusahaan dapat mengoptimalkan rantai pasok, mengurangi risiko kehabisan stok, dan meningkatkan efisiensi operasional gudang.

## Langkah-langkah Utama

1. **Data Preparation & Cleaning**  
   Melakukan pembersihan data, konversi tanggal, dan pembuatan fitur lead time (dalam satuan hari).
2. **Encoding Fitur Kategorikal**  
   Menggunakan LabelEncoder untuk mengubah fitur kategorikal (supplier, lokasi, jenis barang) menjadi numerik.
3. **Pemodelan**  
   Melatih model regresi linear untuk memprediksi lead time berdasarkan supplier, lokasi, jenis barang, dan jumlah order.
4. **Evaluasi**  
   Menghitung metrik akurasi model seperti MAE dan RMSE.
5. **Visualisasi & Insight**  
   Membuat grafik distribusi lead time, rata-rata per supplier, serta scatter plot prediksi vs aktual.

## Hasil Evaluasi Model

- **MAE (Mean Absolute Error):** ~1 hari
- **RMSE (Root Mean Squared Error):** ~1.5 hari  
  Model memiliki akurasi yang cukup baik, dengan rata-rata error prediksi sekitar 1 hari dari nilai aktual.

## Interpretasi & Insight Bisnis

- **Akurasi Model:**  
  Model dapat membantu memprediksi waktu tunggu barang dengan cukup akurat. Ini penting untuk menghindari kekurangan stok dan mengoptimalkan pemesanan.

- **Supplier dengan Lead Time Stabil:**  
  Supplier seperti **Karunia Baru** menunjukkan lead time yang stabil dan dapat diandalkan, sehingga disarankan menjadi prioritas pemesanan untuk barang dengan kebutuhan mendesak.

- **Supplier/Barang dengan Lead Time Tinggi:**  
  Barang seperti **Kompresor** dan supplier **Tiga Dara Elektronik** tercatat memiliki variasi lead time yang lebih tinggi. Disarankan menambah buffer stok atau melakukan pemesanan lebih awal untuk jenis ini.

- **Optimasi Stok dan Proses Bisnis:**  
  Dengan informasi prediksi lead time, perusahaan dapat menyusun reorder point yang lebih presisi, mengurangi risiko kehabisan stok, serta mengoptimalkan cashflow barang.

- **Prioritas Evaluasi:**  
  Supplier dengan variasi lead time tinggi perlu dievaluasi performanya secara berkala, dan jika memungkinkan mencari alternatif supplier yang lebih konsisten.

## Cara Menjalankan

1. Pastikan semua dependencies sudah terpasang (`pandas`, `scikit-learn`, `joblib`, `matplotlib`).
2. Jalankan file utama notebook/script (`data_barang_elektronik.py`).
3. Model dan encoder akan otomatis tersimpan untuk digunakan prediksi data baru.
4. Lihat hasil evaluasi dan visualisasi di output script.

## Kontak

- [minyong007](https://github.com/minyong007)

---
