# Koding_disertasi
Ini adalah Koding Disertasi yang masih belum oke, namun sudah bisa mendapatkan data yang akurat menggunakan dataset keagle
#python kodeFL_disertasi.py 2>&1 | tee output.log
============================================================
 FEDERATED LEARNING DENGAN DYNAMIC TRUST SCORE AGGREGATION
 Implementasi sesuai Disertasi Bab III — Rikie Kartadie
 NIM: 25052050012 | UNY 2025

 Algoritma  : DTS (Dynamic Trust Score)
 
 Versi      : mbuh
 ============================================================

# FORMULA YANG DIIMPLEMENTASIKAN (dari Bab III):
 Persamaan (3.1) — Agregasi Global:
   w^{t+1} = Σ α_k^t · w_k^t
   α_k^t = (T_k^t · n_k) / Σ(T_i^t · n_i)

 Persamaan (3.2) — Dynamic Trust Score:
   T_k^t = β1·Acc̃_k^t + β2·ΔL̃_k^t + β3·(1 − L̃_k^t)
   β1=0.4, β2=0.4, β3=0.2  (β1+β2+β3=1)

 Persamaan (3.3) — Normalisasi Akurasi:
   Acc̃_k^t = (Acc_k^t - min_i Acc_i^t) / (max_i Acc_i^t - min_i Acc_i^t + ε)

 Persamaan (3.4) — Normalisasi Penurunan Loss:
   ΔL̃_k^t = (ΔL_k^t - min_i ΔL_i^t) / (max_i ΔL_i^t - min_i ΔL_i^t + ε)
   ΔL_k^t = L_k^{t-1} - L_k^t

 Persamaan (3.5) — Normalisasi Loss:
   L̃_k^t = (L_k^t - min_i L_i^t) / (max_i L_i^t - min_i L_i^t + ε)

 Persamaan (3.6) — Kriteria Konvergensi:
   |L^t_global - L^{t-1}_global| < δ selama τ round berturut-turut
   δ = 1e-4, τ = 5, T_max = 200
 ============================================================

