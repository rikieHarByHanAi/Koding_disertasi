# Implementasi Federated Learning dengan Dynamic Trust Score (DTS) Aggregation

**Disertasi Bab III – Deteksi Serangan DDoS pada Multi-Controller SDN**  
**Penulis:** Rikie Kartadie  
**NIM:** 25052050012 | **Universitas Negeri Yogyakarta (UNY) – 2025**

Repositori ini berisi implementasi kode dari **Algoritma 1 (Dynamic Trust Score)** sesuai dengan Persamaan (3.1) hingga (3.6) pada disertasi. Algoritma ini dirancang untuk mengagregasi model Federated Learning secara dinamis berdasarkan kepercayaan (Trust Score) masing-masing controller SDN.

---

## 📋 Deskripsi Singkat

Kode ini melakukan simulasi Federated Learning dengan **5 controller SDN** (klien) yang mendistribusikan data serangan DDoS secara **Non-IID** menggunakan distribusi Dirichlet. 
Metode yang diusulkan (DTS) membandingkan bobot agregasi berdasarkan kombinasi:
- **Akurasi lokal** (`Acc̃`)
- **Penurunan Loss** (`ΔL̃`)
- **Loss sesaat** (`L̃`)

Sebagai pembanding, kode ini juga menyertakan baseline **FedAvg (McMahan et al., 2017)** dan **Centralized Training**.

---

## 💻 Spesifikasi Perangkat Keras yang Digunakan

Kode ini telah diuji dan berjalan dengan baik pada spek berikut (tanpa GPU):

| Komponen | Spesifikasi |
| :--- | :--- |
| **Perangkat** | Lenovo ThinkPad T580 |
| **Prosesor (CPU)** | Intel Core i7 (8 Core) |
| **Memori (RAM)** | 16 GB |
| **Akselerasi** | **Tidak memerlukan GPU / CUDA** (hanya mengandalkan CPU) |

> ⚠️ **Catatan:** Karena tidak menggunakan GPU, pelatihan 200 round (T_max sesuai dengan lieratur Karimireddy et al. (2019)) dengan 5 klien mungkin memakan waktu **cukup lama (30–60 menit)** tergantung ukuran dataset. Pastikan laptop terhubung ke listrik.

---

## 📦 Persyaratan Sistem (Dependensi)

Pastikan Python versi **3.8 – 3.11** telah terinstal. Instal semua dependensi yang diperlukan:

```bash
pip install tensorflow kagglehub pandas numpy matplotlib seaborn scikit-learn -q
```
🚀 Cara Menjalankan Program
Clone atau unduh file kodeFL_disertasi.py ke dalam direktori lokal Anda.

Buka terminal / command prompt.

Jalankan perintah berikut (sesuai dengan header kode Anda):

```bash
python kodeFL_disertasi.py 2>&1 | tee output.log
```
Penjelasan perintah:

```bash
python kodeFL_disertasi.py
``` 
: Menjalankan skrip utama.

```bash
2>&1
```
: Menggabungkan output error (stderr) ke output standar (stdout) agar semuanya tercatat.
dan
```bash
| tee output.log
```
: Menampilkan output di layar sekaligus menyimpannya ke file output.log untuk dokumentasi.

## ⚙️ Konfigurasi Hyperparameter (Sesuai Disertasi)
Anda dapat mengubah parameter di bagian KONFIGURASI pada awal skrip:

| Parameter	| Simbol	| Nilai Default	| Keterangan
| :--- | :--- | :--- | :--- |
|Jumlah Controller	| N_CLIENTS	|5	| Jumlah klien SDN |
|Maksimum Round |	T_MAX	| 200	| Batas iterasi global |
|Epoch Lokal |	LOCAL_EPOCHS	| 5	| Pelatihan lokal per klien per round|
|Batch Size	| BATCH_SIZE	| 64	| Ukuran batch tiap iterasi |
|Bobot Akurasi	| BETA_1	| 0.4	| Kontribusi akurasi terhadap Trust (β1)|
|Bobot Penurunan Loss |	BETA_2	| 0.4	| Kontribusi ΔLoss terhadap Trust (β2) |
|Bobot Loss	| BETA_3	| 0.2	| Kontribusi Loss terhadap Trust (β3)|
|Ambang Konvergensi	|DELTA	| 1e-4	| Perubahan loss minimum (Pers. 3.6)|
|Toleransi Konvergensi	| TAU	| 5	|Jumlah round berturut-turut (Pers. 3.6)|
|Non-IID Dirichlet |	DIRICHLET_ALPHA	| 0.5	| Semakin kecil, semakin heterogen data|

## 📊 Output dan Visualisasi
Setelah menjalankan skrip, Anda akan mendapatkan:

#1. Log Terminal / Console
Progres setiap 5 round, menampilkan:
GlobLoss: Rata-rata loss global (L^t_global)
TrAcc: Akurasi training rata-rata
ValAcc: Akurasi validasi rata-rata
ValF1: F1-Score validasi
T1...T5: Nilai Dynamic Trust Score untuk masing-masing controller
α1...α5: Bobot agregasi untuk masing-masing controller

#2. File Grafik (dtsa_results.png & dtsa_results.pdf)
Skrip secara otomatis menghasilkan 9 subplot dalam satu gambar besar, meliputi:
Konvergensi Loss Global (kriteria berhenti)
Akurasi Training & Validasi DTS
Perubahan Dynamic Trust Score per controller
Perubahan Bobot Agregasi (α) per controller
Perbandingan performa DTS vs FedAvg vs Centralized
Confusion Matrix DTS dan FedAvg
Distribusi Non-IID antar controller
Korelasi Trust Score vs Akurasi per controller

## Dataset
https://www.kaggle.com/datasets/aikenkazin/ddos-sdn-dataset
