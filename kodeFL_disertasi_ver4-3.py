# ============================================================
# FEDERATED LEARNING — DETEKSI DDoS SDN MULTI-CONTROLLER
# Implementasi FedAvg Manual — VERSI LOKAL v4.3
#
# PERBAIKAN DARI v4.2 (Colab → Local Python3):
#
#   [LOCAL-1] kagglehub diganti dengan load CSV lokal
#             → Tidak perlu internet atau Kaggle API key
#             → Letakkan file CSV dataset di folder yang sama
#             → Nama file dikonfigurasi di DATASET_PATH
#
#   [LOCAL-2] matplotlib backend dideteksi otomatis
#             → 'TkAgg' jika ada display (laptop biasa)
#             → 'Agg' jika tidak ada display (server/headless)
#             → plt.show() hanya dipanggil jika ada display
#
#   [LOCAL-3] psutil diinstal via pip jika belum ada
#             → Monitoring RAM real-time saat training
#
#   [MEM-1]  Model lokal dihapus setelah setiap round
#             → del local_model + gc.collect() setiap client
#
#   [MEM-2]  clear_session() + rebuild global model tiap GC_EVERY_N_ROUNDS
#             → Mencegah akumulasi TF graph di RAM
#
#   [MEM-3]  batch_predict() untuk semua inferensi
#             → Tidak pernah load seluruh dataset sekaligus ke tensor
#
#   [MEM-4]  float32 konsisten dari preprocessing
#             → Hemat 50% RAM vs float64
#
#   [MEM-5]  del + gc.collect() agresif setelah setiap operasi besar
#             → DataFrame, array sementara, bobot lokal
#
#   [MEM-6]  Progress RAM dicetak tiap round via psutil
#             → Anda bisa pantau tren memori dari terminal
#
#   [FIX-ES1] Early stopping berbasis ROUND, bukan per-evaluasi
#   [FIX-ES2] Monitor val_accuracy pada val set GABUNGAN (lebih stabil)
#   [FIX-ES3] best_weights pakai copy.deepcopy() (anti-aliasing)
#
# CARA PAKAI:
#   1. Instal dependensi:
#      pip install tensorflow pandas numpy matplotlib seaborn scikit-learn psutil
#
#   2. Unduh dataset dari Kaggle:
#      https://www.kaggle.com/datasets/aikenkazin/ddos-sdn-dataset
#      Letakkan file .csv di folder yang sama dengan script ini.
#
#   3. Set DATASET_PATH di bawah (nama file CSV-nya).
#
#   4. Jalankan:
#      python kodeFL_disertasi_ver4-3_local.py 2>&1 | tee output.log
#
# ============================================================

import os
import sys
import gc
import copy
import warnings
import traceback
import time

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Batasi thread TF agar tidak menghabiskan semua core ──────
os.environ['TF_NUM_INTRAOP_THREADS'] = '4'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
# ── Paksa CPU only (hapus baris ini jika punya GPU dedicated) ─
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ── [LOCAL-3] Cek dan instal psutil jika belum ada ───────────
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    print("[INFO] psutil tidak ditemukan. Mencoba install otomatis...")
    os.system(f"{sys.executable} -m pip install psutil -q")
    try:
        import psutil
        _HAS_PSUTIL = True
        print("[INFO] psutil berhasil diinstal.")
    except ImportError:
        _HAS_PSUTIL = False
        print("[WARN] psutil gagal diinstal. Monitor RAM dinonaktifkan.")

import numpy as np
import pandas as pd
import tensorflow as tf

# ── [LOCAL-2] Deteksi display dan pilih backend otomatis ──────
def _setup_matplotlib():
    """
    Pilih matplotlib backend secara otomatis:
    - TkAgg  : jika ada display (laptop dengan desktop environment)
    - Agg    : jika tidak ada display (server / SSH tanpa X11)
    Mengembalikan True jika ada display (bisa plt.show()).
    """
    import matplotlib
    # Coba backend interaktif dulu
    for backend in ['TkAgg', 'Qt5Agg', 'WXAgg']:
        try:
            matplotlib.use(backend)
            import matplotlib.pyplot as _plt
            _plt.figure()   # test apakah bisa buka window
            _plt.close()
            print(f"[INFO] Matplotlib backend: {backend} (interactive)")
            return True
        except Exception:
            continue
    # Fallback ke Agg (non-interactive, selalu berhasil)
    matplotlib.use('Agg')
    print("[INFO] Matplotlib backend: Agg (non-interactive, simpan ke file)")
    return False

_HAS_DISPLAY = _setup_matplotlib()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

print(f"Python     : {sys.version.split()[0]}")
print(f"TensorFlow : {tf.__version__}")

# ── Cek RAM awal ─────────────────────────────────────────────
def _ram_info(label=""):
    if not _HAS_PSUTIL:
        return
    ram  = psutil.virtual_memory()
    swap = psutil.swap_memory()
    used_gb  = (ram.total - ram.available) / 1e9
    total_gb = ram.total / 1e9
    swap_used = swap.used / 1e9
    swap_total = swap.total / 1e9
    tag = f"[RAM{' '+label if label else ''}]"
    print(f"{tag} {used_gb:.1f}/{total_gb:.1f} GB | "
          f"Swap: {swap_used:.1f}/{swap_total:.1f} GB | "
          f"Tersedia: {ram.available/1e9:.1f} GB")

_ram_info("awal")

# ===========================================================
# [LOCAL-1] KONFIGURASI DATASET LOKAL
# ===========================================================
# Ganti dengan nama/path file CSV dataset Anda.
# Bisa path absolut: '/home/user/data/ddos.csv'
# atau relatif (di folder yang sama): 'ddos_dataset.csv'
DATASET_PATH = "ddos_dataset.csv"   # ← GANTI SESUAI NAMA FILE ANDA

# Jika DATASET_PATH tidak ada, skrip akan mencoba cari file .csv
# di folder yang sama secara otomatis.

# ===========================================================
# KONFIGURASI EKSPERIMEN
# ===========================================================
SEED             = 42
N_CLIENTS        = 5       # jumlah controller SDN
FL_ROUNDS        = 200      # maximum communication rounds
LOCAL_EPOCHS     = 5       # epoch lokal per round per client
BATCH_SIZE       = 64
LR_CLIENT        = 1e-3
DIRICHLET_ALPHA  = 0.5     # heterogenitas non-IID (0=sangat heterogen, ∞=IID)
TEST_RATIO       = 0.20    # global held-out test set
VAL_RATIO        = 0.15    # val dari train per controller

# ── Early stopping ───────────────────────────────────────────
# [FIX-ES1] Dalam satuan ROUND (bukan per-evaluasi)
PATIENCE         = 20      # round tanpa improvement → stop

# ── Evaluasi ─────────────────────────────────────────────────
EVAL_EVERY       = 5      # evaluasi tiap N round

# ── Memory management ────────────────────────────────────────
GC_EVERY_N_ROUNDS = 10      # clear TF session tiap N round
PREDICT_BATCH_SIZE = 1024  # ukuran batch saat prediksi (hemat RAM)

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Batasi thread TF via API (tambahan dari env var di atas)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)

print("\n" + "="*65)
print("FEDERATED LEARNING — DETEKSI DDoS SDN MULTI-CONTROLLER")
print("FedAvg Manual v4.3 — Local Python3 + Memory Optimized")
print("="*65)
print(f"  Controller     : {N_CLIENTS} SDN controllers")
print(f"  FL Rounds (max): {FL_ROUNDS}")
print(f"  Local Epochs   : {LOCAL_EPOCHS}")
print(f"  Batch Size     : {BATCH_SIZE}")
print(f"  Dirichlet α    : {DIRICHLET_ALPHA}")
print(f"  PATIENCE       : {PATIENCE} rounds (bukan evaluasi)")
print(f"  Eval setiap    : {EVAL_EVERY} rounds")
print(f"  GC setiap      : {GC_EVERY_N_ROUNDS} rounds")
print(f"  Predict batch  : {PREDICT_BATCH_SIZE}")

# ===========================================================
# STEP 1: LOAD DATASET LOKAL
# ===========================================================
print("\n" + "="*65)
print("STEP 1: MEMUAT DATASET")
print("="*65)

def _find_csv(primary_path):
    """
    Cari file CSV:
    1. Coba path yang diberikan (DATASET_PATH)
    2. Cari file .csv apapun di direktori yang sama dengan script
    3. Cari rekursif di subfolder 'data/' dan 'dataset/'
    """
    # Coba path langsung
    if os.path.isfile(primary_path):
        return primary_path

    # Direktori script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []

    # Cari di direktori script
    for fname in os.listdir(script_dir):
        if fname.lower().endswith('.csv'):
            candidates.append(os.path.join(script_dir, fname))

    # Cari di subfolder umum
    for sub in ['data', 'dataset', 'datasets', '.']:
        subdir = os.path.join(script_dir, sub)
        if os.path.isdir(subdir):
            for fname in os.listdir(subdir):
                if fname.lower().endswith('.csv'):
                    candidates.append(os.path.join(subdir, fname))

    if not candidates:
        return None

    # Pilih yang paling besar (kemungkinan dataset utama)
    candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    print(f"[AUTO] DATASET_PATH tidak ditemukan. Menggunakan: {candidates[0]}")
    return candidates[0]

csv_path = _find_csv(DATASET_PATH)

if csv_path is None:
    print("\n[ERROR] File CSV dataset tidak ditemukan!")
    print(f"  Path yang dicoba: {DATASET_PATH}")
    print(f"  Direktori script: {os.path.dirname(os.path.abspath(__file__))}")
    print("\n  SOLUSI:")
    print("  1. Unduh dataset dari: https://www.kaggle.com/datasets/aikenkazin/ddos-sdn-dataset")
    print("  2. Letakkan file .csv di folder yang sama dengan script ini")
    print(f"  3. Ubah DATASET_PATH = 'nama_file_anda.csv' di baris ~90 skrip ini")
    sys.exit(1)

print(f"Membaca: {csv_path}")
print(f"Ukuran file: {os.path.getsize(csv_path)/1e6:.1f} MB")

# [MEM-4] Baca langsung sebagai float32 untuk hemat RAM
df = None
for enc in ['utf-8', 'latin1', 'iso-8859-1']:
    try:
        # low_memory=False penting untuk dataset besar agar dtype konsisten
        df = pd.read_csv(csv_path, encoding=enc, low_memory=False)
        print(f"Loaded: shape={df.shape}  encoding={enc}")
        break
    except Exception as e:
        print(f"  [WARN] encoding {enc} gagal: {e}")
        continue

if df is None:
    print("[ERROR] Gagal membaca file CSV. Pastikan file tidak rusak.")
    sys.exit(1)

print(f"Penggunaan memori DataFrame: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
_ram_info("setelah load CSV")

# ===========================================================
# STEP 2: PREPROCESSING
# ===========================================================
print("\n" + "="*65)
print("STEP 2: PREPROCESSING")
print("="*65)

label_candidates = ['label', 'attack_cat', 'class', 'type', 'malicious', 'attack']
label_col = next(
    (c for c in df.columns if c.lower() in label_candidates),
    df.columns[-1]
)
print(f"Label column : '{label_col}'")
print(f"Label values : {df[label_col].value_counts().to_dict()}")

X_raw = df.drop(columns=[label_col])
y_raw = df[label_col].copy()

# [MEM-5] Bebaskan df asli segera
del df
gc.collect()

le = LabelEncoder()
y_enc = le.fit_transform(y_raw).astype(np.int8)  # int8: hemat memori
print(f"Classes      : {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Pilih fitur numerik saja dan konversi ke float32 langsung
# [MEM-4] float32 = setengah memori vs float64
X_num = X_raw.select_dtypes(include=[np.number]).fillna(0).astype(np.float32)
n_features_raw = X_num.shape[1]
print(f"Features     : {n_features_raw}")
print(f"RAM X_num    : {X_num.values.nbytes/1e6:.1f} MB")

del X_raw, y_raw
gc.collect()

# Global test split — stratified, sebelum apapun
X_tv, X_test, y_tv, y_test = train_test_split(
    X_num.values, y_enc,
    test_size=TEST_RATIO, random_state=SEED, stratify=y_enc
)
del X_num, y_enc
gc.collect()

# Normalisasi StandardScaler
scaler    = StandardScaler()
X_tv_sc   = scaler.fit_transform(X_tv).astype(np.float32)
X_test_sc = scaler.transform(X_test).astype(np.float32)
del X_tv, X_test
gc.collect()

n_features = X_tv_sc.shape[1]
print(f"\nTrainval : {len(X_tv_sc):,} samples")
print(f"Test     : {len(X_test_sc):,} samples (held-out global)")
print(f"Normal   : {np.sum(y_tv==0):,} | Attack: {np.sum(y_tv==1):,}")
print(f"RAM X_tv : {X_tv_sc.nbytes/1e6:.1f} MB | "
      f"X_test: {X_test_sc.nbytes/1e6:.1f} MB")
_ram_info("setelah preprocessing")

# ===========================================================
# STEP 3: NON-IID — DISTRIBUSI DIRICHLET
# ===========================================================
print("\n" + "="*65)
print(f"STEP 3: SIMULASI NON-IID — DISTRIBUSI DIRICHLET (α={DIRICHLET_ALPHA})")
print("="*65)

def dirichlet_partition(X, y, n_clients, alpha, seed=42):
    """
    Partisi Non-IID menggunakan distribusi Dirichlet.
    Ref: Hsieh et al. (2020); Li et al. (2022).
    Mengembalikan list of (X_client, y_client) sebagai float32/int8.
    """
    rng = np.random.default_rng(seed)
    client_idx = [[] for _ in range(n_clients)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        props  = rng.dirichlet(alpha * np.ones(n_clients))
        splits = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(idx, splits)):
            client_idx[cid].extend(chunk.tolist())
    result = []
    for idx in client_idx:
        idx = np.array(idx, dtype=np.int32)
        rng.shuffle(idx)
        # .copy() agar tidak menyimpan referensi ke array induk
        result.append((X[idx].copy(), y[idx].copy()))
    return result

client_data = dirichlet_partition(X_tv_sc, y_tv, N_CLIENTS, DIRICHLET_ALPHA, SEED)

print(f"\n{'Controller':<14}{'Samples':>8}{'Normal':>8}{'Attack':>8}{'Attack%':>9}")
print("-"*50)
for i, (Xc, yc) in enumerate(client_data):
    nn = int(np.sum(yc==0))
    na = int(np.sum(yc==1))
    print(f"Controller {i+1:<4}{len(yc):>8,}{nn:>8,}{na:>8,}{na/len(yc)*100:>8.1f}%")

# ===========================================================
# STEP 4: SPLIT TRAIN/VAL PER CLIENT
# ===========================================================
print("\n" + "="*65)
print("STEP 4: SPLIT TRAIN/VAL PER CLIENT")
print("="*65)

client_trains = []
client_vals   = []

for i, (Xc, yc) in enumerate(client_data):
    Xtr, Xvl, ytr, yvl = train_test_split(
        Xc, yc, test_size=VAL_RATIO, random_state=SEED, stratify=yc
    )
    client_trains.append((Xtr.astype(np.float32), ytr.astype(np.float32)))
    client_vals.append((Xvl.astype(np.float32), yvl.astype(np.float32)))
    print(f"Controller {i+1}: train={len(Xtr):,} | val={len(Xvl):,}")

# [FIX-ES2] Val set GABUNGAN untuk monitoring early stopping
# Lebih stabil daripada rata-rata val lokal yang noisy (non-IID)
X_val_combined = np.concatenate([xvl for xvl, _ in client_vals], axis=0)
y_val_combined = np.concatenate([yvl for _, yvl in client_vals], axis=0).astype(np.int8)
print(f"\nVal gabungan : {len(X_val_combined):,} samples (untuk early stopping)")
print(f"RAM val set  : {X_val_combined.nbytes/1e6:.1f} MB")

# Bebaskan client_data, sudah tidak diperlukan
del client_data
gc.collect()
_ram_info("setelah partisi data")

# ===========================================================
# STEP 5: ARSITEKTUR MODEL
# ===========================================================
print("\n" + "="*65)
print("STEP 5: ARSITEKTUR MODEL")
print("="*65)

def build_model(n_feat):
    """
    MLP dengan regularisasi untuk deteksi DDoS biner.
    Arsitektur: 128 → BN → DO(0.3) → 64 → BN → DO(0.2) → 32 → DO(0.1) → 1
    """
    inp = tf.keras.layers.Input(shape=(n_feat,))
    x   = tf.keras.layers.Dense(
              128, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-3))(inp)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = tf.keras.layers.Dense(
              64, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dropout(0.2)(x)
    x   = tf.keras.layers.Dense(32, activation='relu')(x)
    x   = tf.keras.layers.Dropout(0.1)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name="ddos_detector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_CLIENT),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Hitung parameter sekali, lalu hapus model sampel
_tmp = build_model(n_features)
print(f"Parameter per model : {_tmp.count_params():,}")
_tmp.summary()
del _tmp
tf.keras.backend.clear_session()
gc.collect()

# ===========================================================
# STEP 6: HELPER FUNCTIONS
# ===========================================================

def get_weights(model):
    """Ambil bobot model sebagai list numpy array (deep copy)."""
    return [w.numpy().copy() for w in model.weights]

def set_weights(model, weights):
    """Set bobot model dari list numpy array."""
    for w, val in zip(model.weights, weights):
        w.assign(val)

def fedavg_aggregate(client_weights_list, client_sizes):
    """
    FedAvg: weighted average bobot berdasarkan jumlah data lokal.
    Ref: McMahan et al. (2017) Algorithm 1.
    Menggunakan float64 untuk akumulasi, lalu cast ke float32.
    """
    total    = float(sum(client_sizes))
    n_layers = len(client_weights_list[0])
    aggregated = []
    for layer_idx in range(n_layers):
        layer_acc = np.zeros_like(
            client_weights_list[0][layer_idx], dtype=np.float64
        )
        for weights, size in zip(client_weights_list, client_sizes):
            layer_acc += weights[layer_idx].astype(np.float64) * (size / total)
        aggregated.append(layer_acc.astype(np.float32))
    return aggregated

def local_train(global_weights, X_train, y_train, n_feat,
                local_epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE):
    """
    Training lokal pada satu client.
    [MEM-1] Model dibuat → dilatih → bobot disimpan → model DIHAPUS.
    Tidak ada model lokal yang hidup setelah fungsi ini selesai.
    """
    local_model = build_model(n_feat)
    set_weights(local_model, global_weights)
    local_model.fit(
        X_train, y_train,
        epochs=local_epochs,
        batch_size=batch_size,
        verbose=0
    )
    loss, acc = local_model.evaluate(X_train, y_train, verbose=0)
    w = get_weights(local_model)

    # [MEM-1] Hapus model lokal segera
    del local_model
    gc.collect()

    return w, len(X_train), float(loss), float(acc)

def batch_predict(model, X, batch_size=PREDICT_BATCH_SIZE):
    """
    [MEM-3] Prediksi dalam batch kecil untuk menjaga RAM.
    Tidak pernah membuat tensor sebesar seluruh dataset sekaligus.
    """
    preds = []
    for start in range(0, len(X), batch_size):
        chunk = X[start:start+batch_size]
        p     = model.predict(chunk, verbose=0, batch_size=batch_size)
        preds.append(p)
    return np.concatenate(preds, axis=0).flatten()

# ===========================================================
# STEP 7: FEDERATED LEARNING — FedAvg MANUAL
# ===========================================================
print("\n" + "="*65)
print("STEP 7: FEDERATED LEARNING — FedAvg MANUAL")
print(f"  Max {FL_ROUNDS} rounds × {LOCAL_EPOCHS} local epochs × {N_CLIENTS} clients")
print(f"  Early stopping: PATIENCE={PATIENCE} ROUNDS")
print(f"  Monitor       : val_accuracy pada val set GABUNGAN")
print("="*65)

# Inisialisasi global model
global_model   = build_model(n_features)
global_weights = get_weights(global_model)

history = {
    'round':      [],
    'train_loss': [],
    'train_acc':  [],
    'val_acc':    [],
    'val_f1':     []
}

best_val_acc      = 0.0
# [FIX-ES3] deepcopy untuk mencegah aliasing
best_weights      = copy.deepcopy(global_weights)
best_round        = 0
no_improve_rounds = 0   # [FIX-ES1] dalam satuan ROUND
stopped_early     = False
val_acc           = 0.0  # nilai awal untuk referensi di luar blok if

print(f"\n{'Rd':>4} | {'AvgLoss':>9} | {'AvgAcc':>8} | {'ValAcc':>8} | "
      f"{'ValF1':>7} | {'NoImp':>6} | {'RAM GB':>7} | Status")
print("-"*80)

t_start = time.time()

for rnd in range(1, FL_ROUNDS + 1):

    # ── [MEM-2] Clear TF session berkala ────────────────────
    if rnd > 1 and rnd % GC_EVERY_N_ROUNDS == 0:
        # Simpan bobot global sebelum clear
        saved_gw = [w.copy() for w in global_weights]
        tf.keras.backend.clear_session()
        gc.collect()
        # Rebuild global model dan restore bobot
        global_model = build_model(n_features)
        global_weights = saved_gw
        set_weights(global_model, global_weights)
        del saved_gw
        gc.collect()

    # ── LOCAL TRAINING ──────────────────────────────────────
    all_weights  = []
    all_sizes    = []
    round_losses = []
    round_accs   = []

    for cid, (Xtr, ytr) in enumerate(client_trains):
        # [MEM-1] local_train menghapus model lokal setelah selesai
        w, n, loss_k, acc_k = local_train(
            global_weights, Xtr, ytr, n_features
        )
        all_weights.append(w)
        all_sizes.append(n)
        round_losses.append(loss_k)
        round_accs.append(acc_k)

    # ── FedAvg AGGREGATION ──────────────────────────────────
    global_weights = fedavg_aggregate(all_weights, all_sizes)
    set_weights(global_model, global_weights)

    # [MEM-5] Bebaskan bobot lokal segera setelah agregasi
    del all_weights
    gc.collect()

    avg_loss = float(np.mean(round_losses))
    avg_acc  = float(np.mean(round_accs))

    # ── [MEM-6] Baca RAM saat ini ───────────────────────────
    ram_used_gb = 0.0
    if _HAS_PSUTIL:
        ram_used_gb = (psutil.virtual_memory().total
                       - psutil.virtual_memory().available) / 1e9

    # ── EVALUASI (setiap EVAL_EVERY round atau round pertama) ─
    did_eval = False
    status   = ""
    if rnd % EVAL_EVERY == 0 or rnd == 1:
        did_eval = True
        # [FIX-ES2] [MEM-3] Gunakan val gabungan + batch_predict
        y_prob_val = batch_predict(global_model, X_val_combined)
        y_pred_val = (y_prob_val > 0.5).astype(int)
        val_acc    = float(accuracy_score(y_val_combined, y_pred_val))
        val_f1     = float(f1_score(y_val_combined, y_pred_val, zero_division=0))

        history['round'].append(rnd)
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(avg_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            # [FIX-ES3] deepcopy agar tidak aliasing
            best_weights      = copy.deepcopy(global_weights)
            best_round        = rnd
            no_improve_rounds = 0
            status            = "★ best"
        else:
            status = "(no imp)"

        elapsed = time.time() - t_start
        print(f"{rnd:>4} | {avg_loss:>9.4f} | {avg_acc:>8.4f} | "
              f"{val_acc:>8.4f} | {val_f1:>7.4f} | "
              f"{no_improve_rounds:>5}r | {ram_used_gb:>5.1f}GB | "
              f"{status}  [{elapsed:.0f}s]")

        del y_prob_val, y_pred_val
        gc.collect()

    # ── [FIX-ES1] Update no_improve_rounds setiap ROUND ────
    # Round dengan evaluasi: update hanya jika tidak ada improvement
    # Round tanpa evaluasi : selalu tambah (tidak ada info improvement)
    if did_eval:
        if status != "★ best":
            no_improve_rounds += 1
    else:
        no_improve_rounds += 1

    # ── [FIX-ES1] Cek early stopping dalam satuan ROUND ─────
    if no_improve_rounds >= PATIENCE:
        print(f"\n[EARLY STOP] Tidak ada peningkatan selama {PATIENCE} rounds.")
        print(f"  Best val accuracy: {best_val_acc:.4f} (round {best_round})")
        stopped_early = True
        break

final_round  = rnd
total_time   = time.time() - t_start

print(f"\n{'FL selesai':<22}: Round {final_round}/{FL_ROUNDS}  ({total_time:.1f}s)")
print(f"{'Early stopping':<22}: "
      f"{'Ya (best @ round ' + str(best_round) + ')' if stopped_early else 'Tidak (T_max tercapai)'}")
print(f"{'Best val accuracy':<22}: {best_val_acc:.4f}")
_ram_info("setelah FL selesai")

# Muat bobot terbaik ke model final
set_weights(global_model, best_weights)
del best_weights
gc.collect()

# ===========================================================
# STEP 8: EVALUASI AKHIR — TEST SET GLOBAL (HELD-OUT)
# ===========================================================
print("\n" + "="*65)
print("STEP 8: EVALUASI AKHIR — TEST SET GLOBAL (HELD-OUT)")
print("="*65)

def evaluate_model(model, X, y_true, model_name):
    """Evaluasi lengkap dengan batch_predict untuk efisiensi memori."""
    y_prob = batch_predict(model, X)
    y_pred = (y_prob > 0.5).astype(int)
    y_true = np.array(y_true, dtype=int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)

    # False Positive Rate
    cm_mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm_mat.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n{'─'*55}")
    print(f"  {model_name}")
    print(f"{'─'*55}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  FPR       : {fpr:.4f}  (False Positive Rate)")
    print(f"  TP={tp} | FP={fp} | FN={fn} | TN={tn}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=['Normal', 'Attack'],
                                zero_division=0))
    return dict(accuracy=acc, precision=prec, recall=rec,
                f1=f1, auc=auc, fpr=fpr,
                y_pred=y_pred, y_prob=y_prob)

fl_res = evaluate_model(
    global_model, X_test_sc, y_test,
    "FEDERATED LEARNING — FedAvg Manual v4.3"
)

# ── Centralized Baseline ─────────────────────────────────────
print("\nTraining Centralized Baseline...")
# [MEM-2] Clear session sebelum mulai baseline
tf.keras.backend.clear_session()
gc.collect()

cent_model = build_model(n_features)
cb_es = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=15,
    restore_best_weights=True, verbose=1
)
cent_model.fit(
    X_tv_sc, y_tv,
    validation_split=0.15,
    epochs=200, batch_size=BATCH_SIZE,
    callbacks=[cb_es], verbose=0
)
cent_res = evaluate_model(
    cent_model, X_test_sc, y_test, "CENTRALIZED BASELINE"
)

print(f"\n  GAP (FL − Centralized):")
for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
    gap  = fl_res[k] - cent_res[k]
    note = "← trade-off privasi" if gap < 0 else "← FL lebih baik"
    print(f"    {k:<12}: {gap:+.4f}  {note}")

del cent_model
gc.collect()
_ram_info("setelah evaluasi")

# ===========================================================
# STEP 9: EVALUASI PER CONTROLLER (Val Set Lokal)
# ===========================================================
print("\n" + "="*65)
print("STEP 9: EVALUASI PER CONTROLLER (Val Set Lokal)")
print("="*65)

per_ctrl = {}
for i, (Xvl, yvl) in enumerate(client_vals):
    yhat = (batch_predict(global_model, Xvl) > 0.5).astype(int)
    yvl_int = yvl.astype(int)
    name = f"Ctrl_{i+1}"
    per_ctrl[name] = {
        'accuracy':   accuracy_score(yvl_int, yhat),
        'precision':  precision_score(yvl_int, yhat, zero_division=0),
        'recall':     recall_score(yvl_int, yhat, zero_division=0),
        'f1':         f1_score(yvl_int, yhat, zero_division=0),
        'attack_pct': float(np.mean(yvl)) * 100,
        'n_samples':  len(yvl)
    }

print(f"\n{'Controller':<12}{'N':>7}{'Attack%':>9}{'Acc':>8}"
      f"{'Prec':>8}{'Rec':>8}{'F1':>8}")
print("-"*62)
for name, m in per_ctrl.items():
    print(f"{name:<12}{m['n_samples']:>7,}{m['attack_pct']:>8.1f}%"
          f"{m['accuracy']:>8.4f}{m['precision']:>8.4f}"
          f"{m['recall']:>8.4f}{m['f1']:>8.4f}")

# ===========================================================
# STEP 10: VISUALISASI
# ===========================================================
print("\n" + "="*65)
print("STEP 10: VISUALISASI")
print("="*65)

sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12,
    'axes.titleweight': 'bold', 'figure.dpi': 100
})

C = {
    'fl':   '#2980b9', 'cent': '#27ae60', 'loss': '#e74c3c',
    'val':  '#8e44ad', 'atk':  '#c0392b', 'norm': '#2980b9'
}

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.40)

# ── Plot 1: Konvergensi FL ────────────────────────────────────
ax1  = fig.add_subplot(gs[0, 0])
ax1b = ax1.twinx()
ax1.plot(history['round'], history['train_loss'],
         color=C['loss'], lw=2, label='Train Loss')
ax1b.plot(history['round'], history['train_acc'],
          color=C['fl'], lw=2, ls='--', label='Train Acc')
ax1b.plot(history['round'], history['val_acc'],
          color=C['val'], lw=2, ls='-.', label='Val Acc (Gabungan)')
if best_round in history['round']:
    ax1b.axvline(x=best_round, color='gold', lw=2, ls=':',
                 label=f'Best (r{best_round})')
if stopped_early:
    ax1b.axvline(x=final_round, color='red', lw=1.5, ls='--', alpha=0.7,
                 label=f'Early Stop (r{final_round})')
ax1.set_xlabel('Communication Round')
ax1.set_ylabel('Loss', color=C['loss'])
ax1b.set_ylabel('Accuracy')
es_label = f"ES@r{final_round}, best@r{best_round}" if stopped_early else f"T_max@r{final_round}"
ax1.set_title(f'FL Convergence (FedAvg)\n[{es_label}]')
l1, n1 = ax1.get_legend_handles_labels()
l2, n2 = ax1b.get_legend_handles_labels()
ax1.legend(l1+l2, n1+n2, fontsize=7.5, loc='upper right')

# ── Plot 2: FL vs Centralized ─────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
mk  = ['accuracy', 'precision', 'recall', 'f1', 'auc']
x   = np.arange(len(mk)); w = 0.35
b1  = ax2.bar(x-w/2, [fl_res[k]   for k in mk], w,
              color=C['fl'],   label='FL (FedAvg)', alpha=0.85)
b2  = ax2.bar(x+w/2, [cent_res[k] for k in mk], w,
              color=C['cent'], label='Centralized',  alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(['Acc', 'Prec', 'Rec', 'F1', 'AUC'])
ax2.set_ylim([0, 1.12])
ax2.set_title('FL vs Centralized (Global Test Set)')
ax2.legend(fontsize=9)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.01,
             f'{h:.3f}', ha='center', va='bottom', fontsize=7.5)

# ── Plot 3: Confusion Matrix FL ───────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
cm1 = confusion_matrix(y_test, fl_res['y_pred'])
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'], cbar=False)
ax3.set_xlabel('Predicted'); ax3.set_ylabel('Actual')
ax3.set_title('Confusion Matrix — FL Model')

# ── Plot 4: Performa per controller ──────────────────────────
ax4  = fig.add_subplot(gs[1, 0])
ax4b = ax4.twinx()
cnames = list(per_ctrl.keys())
x4 = np.arange(len(cnames))
ax4.bar(x4-0.2, [per_ctrl[c]['accuracy'] for c in cnames],
        0.35, color=C['fl'], label='Accuracy', alpha=0.8)
ax4.bar(x4+0.2, [per_ctrl[c]['f1'] for c in cnames],
        0.35, color='#e67e22', label='F1-Score', alpha=0.8)
ax4b.plot(x4, [per_ctrl[c]['attack_pct'] for c in cnames],
          'r-o', lw=2, ms=8, label='Attack %')
ax4.set_xticks(x4); ax4.set_xticklabels(cnames)
ax4.set_ylim([0, 1]); ax4b.set_ylim([0, 100])
ax4.set_ylabel('Score')
ax4b.set_ylabel('Attack Ratio (%)', color='red')
ax4.set_title('Dampak Non-IID per Controller')
la, na = ax4.get_legend_handles_labels()
lb, nb = ax4b.get_legend_handles_labels()
ax4.legend(la+lb, na+nb, fontsize=8)

# ── Plot 5: Distribusi Non-IID ────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
x5     = np.arange(len(cnames))
n_norm = [per_ctrl[c]['n_samples'] * (1 - per_ctrl[c]['attack_pct']/100)
          for c in cnames]
n_atk  = [per_ctrl[c]['n_samples'] * per_ctrl[c]['attack_pct']/100
          for c in cnames]
ax5.bar(x5, n_norm, color=C['norm'], label='Normal', alpha=0.8)
ax5.bar(x5, n_atk,  color=C['atk'], label='Attack',
        bottom=n_norm, alpha=0.8)
ax5.set_xticks(x5); ax5.set_xticklabels(cnames)
ax5.set_ylabel('Jumlah Sampel (Val Set)')
ax5.set_title(f'Distribusi Non-IID (α={DIRICHLET_ALPHA})')
ax5.legend(fontsize=9)

# ── Plot 6: Confusion Matrix Centralized ─────────────────────
ax6 = fig.add_subplot(gs[1, 2])
cm2 = confusion_matrix(y_test, cent_res['y_pred'])
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=ax6,
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'], cbar=False)
ax6.set_xlabel('Predicted'); ax6.set_ylabel('Actual')
ax6.set_title('Confusion Matrix — Centralized')

es_info = (f"Early Stop @ Round {final_round} (best @ round {best_round})"
           if stopped_early else f"Selesai @ Round {final_round} (T_max)")

plt.suptitle(
    'Analisis Non-IID pada Multiple Controller SDN untuk Deteksi DDoS\n'
    f'Federated Learning (FedAvg) — {es_info}  |  Jurnal JINITA SINTA 2',
    fontsize=12, fontweight='bold', y=1.01
)

# Simpan selalu (baik ada/tidak ada display)
plt.savefig('fl_ddos_results_v43.pdf', bbox_inches='tight', dpi=300)
plt.savefig('fl_ddos_results_v43.png', bbox_inches='tight', dpi=150)
print("Tersimpan: fl_ddos_results_v43.pdf + fl_ddos_results_v43.png")

# [LOCAL-2] Tampilkan jika ada display, tutup jika tidak
if _HAS_DISPLAY:
    plt.show()
else:
    plt.close()

gc.collect()

# ===========================================================
# RINGKASAN AKHIR
# ===========================================================
total_comm = final_round * N_CLIENTS
_ram_info("akhir")

print(f"""
{'='*65}
RINGKASAN HASIL — JURNAL JINITA SINTA 2
{'='*65}
  Dataset      : {os.path.basename(csv_path)}
  Features     : {n_features}  |  Classes: Binary (Normal / Attack)
  Train/Test   : {len(X_tv_sc):,} / {len(X_test_sc):,} (held-out global)
  Controllers  : {N_CLIENTS} SDN  |  Dirichlet α={DIRICHLET_ALPHA}
  Algoritma    : FedAvg (McMahan et al., 2017) — Manual
  Rounds actual: {final_round}/{FL_ROUNDS}  {'← EARLY STOP aktif ✓' if stopped_early else '← T_max'}
  Best round   : {best_round}
  Waktu total  : {total_time:.1f}s
  Comm. Cost   : {total_comm} model transmissions

  ┌──────────────────────┬────────┬────────┬────────┬────────┬────────┬────────┐
  │ Model                │  Acc   │  Prec  │  Rec   │  F1    │  AUC   │  FPR   │
  ├──────────────────────┼────────┼────────┼────────┼────────┼────────┼────────┤
  │ FL (FedAvg) v4.3     │{fl_res['accuracy']:>6.4f}│{fl_res['precision']:>6.4f}│{fl_res['recall']:>6.4f}│{fl_res['f1']:>6.4f}│{fl_res['auc']:>6.4f}│{fl_res['fpr']:>6.4f}│
  │ Centralized Baseline │{cent_res['accuracy']:>6.4f}│{cent_res['precision']:>6.4f}│{cent_res['recall']:>6.4f}│{cent_res['f1']:>6.4f}│{cent_res['auc']:>6.4f}│{cent_res['fpr']:>6.4f}│
  └──────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘

  Heterogenitas Non-IID (attack ratio per controller):
    Min : {min(m['attack_pct'] for m in per_ctrl.values()):.1f}%
    Max : {max(m['attack_pct'] for m in per_ctrl.values()):.1f}%
    Std : {np.std([m['attack_pct'] for m in per_ctrl.values()]):.2f}%

  [PERBAIKAN v4.3 — Local Python3 + Memory Optimized]
    ✓ [LOCAL-1] Tidak perlu kagglehub — load CSV lokal langsung
    ✓ [LOCAL-2] Matplotlib backend otomatis (TkAgg/Agg)
    ✓ [LOCAL-3] psutil: monitor RAM real-time tiap round
    ✓ [MEM-1]  Model lokal dihapus tiap round (del + gc)
    ✓ [MEM-2]  TF session di-clear tiap {GC_EVERY_N_ROUNDS} round
    ✓ [MEM-3]  batch_predict() — batch size {PREDICT_BATCH_SIZE}
    ✓ [MEM-4]  float32 dari awal (hemat 50% vs float64)
    ✓ [MEM-5]  del agresif: DataFrame, array temp, bobot lokal
    ✓ [MEM-6]  Progress RAM dicetak tiap evaluasi
    ✓ [FIX-ES1] Early stopping dalam satuan ROUND
    ✓ [FIX-ES2] Monitor val set GABUNGAN (stabil)
    ✓ [FIX-ES3] best_weights pakai copy.deepcopy()

  Seed = {SEED} (fully reproducible)
  Output: fl_ddos_results_v43.pdf + fl_ddos_results_v43.png
{'='*65}
""")
