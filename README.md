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

