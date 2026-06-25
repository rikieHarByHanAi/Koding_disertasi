#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEDERATED LEARNING DENGAN DYNAMIC TRUST SCORE AGGREGATION
Implementasi sesuai Disertasi Bab III — Rikie Kartadie
NIM: 25052050012 | UNY 2025

Algoritma  : DTS (Dynamic Trust Score)

CARA MENJALANKAN:
    python3 kodeFL_disertasi_01.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Import kagglehub dengan error handling
try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("⚠️ kagglehub tidak terinstall. Install dengan: pip install kagglehub")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

print(f"Python     : {sys.version.split()[0]}")
print(f"TensorFlow : {tf.__version__}")

# ===========================================================
# SEED GLOBAL
# ===========================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ===========================================================
# KONFIGURASI — sesuai Bab III Disertasi
# ===========================================================
BETA_1 = 0.4
BETA_2 = 0.4
BETA_3 = 0.2
N_CLIENTS = 5
T_MAX = 200
LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LR_CLIENT = 1e-3
DELTA = 1e-4
TAU = 5
PATIENCE = 20
EPSILON = 1e-8
DIRICHLET_ALPHA = 0.5
TEST_RATIO = 0.20

print("\n" + "="*65)
print("DTS: DYNAMIC TRUST SCORE")
print("Implementasi sesuai Disertasi Bab III — Rikie Kartadie")
print("="*65)
print(f"  Controller  : {N_CLIENTS}  |  T_max : {T_MAX}")
print(f"  Local Epochs: {LOCAL_EPOCHS}  |  Batch  : {BATCH_SIZE}")
print(f"  β1={BETA_1}, β2={BETA_2}, β3={BETA_3}")
print(f"  δ={DELTA}, τ={TAU}  |  ε={EPSILON}")
print(f"  Patience    : {PATIENCE} rounds")

# ===========================================================
# STEP 1: LOAD DATASET
# ===========================================================
print("\n" + "="*65)
print("STEP 1: MEMUAT DATASET")
print("="*65)

df = None

# Method 1: Coba dari kagglehub
if KAGGLE_AVAILABLE:
    try:
        path = kagglehub.dataset_download("aikenkazin/ddos-sdn-dataset")
        print(f"Kaggle path: {path}")
        
        for root, _, files in os.walk(path):
            for fname in files:
                if fname.endswith('.csv'):
                    fpath = os.path.join(root, fname)
                    for enc in ['utf-8', 'latin1']:
                        try:
                            df = pd.read_csv(fpath, encoding=enc)
                            print(f"Loaded: {fname}  shape={df.shape}")
                            break
                        except Exception:
                            continue
                    if df is not None:
                        break
            if df is not None:
                break
    except Exception as e:
        print(f"Kaggle download error: {e}")
        df = None

# Method 2: Cari dataset lokal
if df is None:
    print("Mencari dataset lokal di direktori saat ini...")
    for fname in os.listdir('.'):
        if fname.endswith('.csv'):
            try:
                df = pd.read_csv(fname, encoding='utf-8')
                print(f"Loaded local: {fname} shape={df.shape}")
                break
            except:
                try:
                    df = pd.read_csv(fname, encoding='latin1')
                    print(f"Loaded local: {fname} shape={df.shape}")
                    break
                except:
                    continue

if df is None:
    print("\n❌ ERROR: Dataset tidak ditemukan!")
    print("\nSolusi:")
    print("1. Pastikan koneksi internet aktif")
    print("2. Install kagglehub: pip install kagglehub")
    print("3. Atau download manual dataset dari:")
    print("   https://www.kaggle.com/datasets/aikenkazin/ddos-sdn-dataset")
    print("   dan letakkan file CSV di folder yang sama dengan script ini")
    sys.exit(1)

# ===========================================================
# STEP 2: PREPROCESSING
# ===========================================================
print("\n" + "="*65)
print("STEP 2: PREPROCESSING")
print("="*65)

label_candidates = ['label', 'attack_cat', 'class', 'type', 'malicious', 'attack']
label_col = None
for col in df.columns:
    if col.lower() in label_candidates:
        label_col = col
        break

if label_col is None:
    label_col = df.columns[-1]
    print(f"⚠️ Menggunakan kolom terakhir sebagai label: {label_col}")

print(f"Label column : '{label_col}'")

X_raw = df.drop(columns=[label_col])
y_raw = df[label_col]

le = LabelEncoder()
y_enc = le.fit_transform(y_raw)
print(f"Classes      : {dict(zip(le.classes_, le.transform(le.classes_)))}")

X_num = X_raw.select_dtypes(include=[np.number]).fillna(0)
print(f"Features     : {X_num.shape[1]}")

# Held-out test set
X_tv, X_test, y_tv, y_test = train_test_split(
    X_num.values, y_enc,
    test_size=TEST_RATIO, random_state=SEED, stratify=y_enc
)

scaler = StandardScaler()
X_tv_sc = scaler.fit_transform(X_tv)
X_test_sc = scaler.transform(X_test)

n_features = X_tv_sc.shape[1]
print(f"\nTrainval : {len(X_tv_sc):,} | Test: {len(X_test_sc):,} (held-out)")
print(f"Normal   : {np.sum(y_tv==0):,} | Attack: {np.sum(y_tv==1):,}")

# ===========================================================
# STEP 3: NON-IID DISTRIBUSI DIRICHLET
# ===========================================================
print("\n" + "="*65)
print(f"STEP 3: DISTRIBUSI NON-IID (DIRICHLET α={DIRICHLET_ALPHA})")
print("="*65)

def dirichlet_partition(X, y, n_clients, alpha, seed=42):
    rng = np.random.default_rng(seed)
    client_idx = [[] for _ in range(n_clients)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        props = rng.dirichlet(alpha * np.ones(n_clients))
        splits = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(idx, splits)):
            client_idx[cid].extend(chunk.tolist())
    result = []
    for idx in client_idx:
        idx = np.array(idx, dtype=int)
        rng.shuffle(idx)
        result.append((X[idx], y[idx]))
    return result

client_data = dirichlet_partition(X_tv_sc, y_tv, N_CLIENTS, DIRICHLET_ALPHA, SEED)

print(f"\n{'Controller':<14}{'Samples':>8}{'Normal':>8}{'Attack':>8}{'Attack%':>9}")
print("-"*50)
for i, (Xc, yc) in enumerate(client_data):
    nn = int(np.sum(yc==0))
    na = int(np.sum(yc==1))
    print(f"Controller {i+1:<4}{len(yc):>8,}{nn:>8,}{na:>8,}{na/len(yc)*100:>8.1f}%")

# Split train/val per-client
client_trains, client_vals = [], []
for Xc, yc in client_data:
    Xtr, Xvl, ytr, yvl = train_test_split(
        Xc, yc, test_size=0.15, random_state=SEED, stratify=yc
    )
    client_trains.append((Xtr, ytr))
    client_vals.append((Xvl, yvl))

# ===========================================================
# STEP 4: ARSITEKTUR MODEL
# ===========================================================
print("\n" + "="*65)
print("STEP 4: ARSITEKTUR MODEL")
print("="*65)

def build_model(n_feat):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_feat,)),
        tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name="ddos_detector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_CLIENT),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

sample = build_model(n_features)
print(f"Parameter: {sample.count_params():,}")
sample.summary()

# ===========================================================
# FUNGSI UTILITAS
# ===========================================================
def get_weights(model):
    return [w.numpy().copy() for w in model.weights]

def set_weights(model, weights):
    for w, val in zip(model.weights, weights):
        w.assign(val)

def local_train(global_weights, X_train, y_train, n_feat):
    local_model = build_model(n_feat)
    set_weights(local_model, global_weights)
    local_model.fit(
        X_train.astype(np.float32), y_train.astype(np.float32),
        epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0
    )
    loss, acc = local_model.evaluate(
        X_train.astype(np.float32), y_train.astype(np.float32), verbose=0
    )
    return get_weights(local_model), len(X_train), float(loss), float(acc)

def minmax_normalize(values, eps=EPSILON):
    arr = np.array(values, dtype=np.float64)
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + eps)

def compute_dts(acc_list, loss_list, prev_loss_list):
    acc_arr = np.array(acc_list, dtype=np.float64)
    loss_arr = np.array(loss_list, dtype=np.float64)
    
    if prev_loss_list is None:
        delta_loss = np.zeros_like(loss_arr)
    else:
        prev_arr = np.array(prev_loss_list, dtype=np.float64)
        delta_loss = prev_arr - loss_arr
    
    acc_norm = minmax_normalize(acc_arr)
    delta_norm = minmax_normalize(delta_loss)
    loss_norm = minmax_normalize(loss_arr)
    
    T = BETA_1 * acc_norm + BETA_2 * delta_norm + BETA_3 * (1.0 - loss_norm)
    return T

def dts_aggregate(client_weights_list, client_sizes, trust_scores):
    T = np.array(trust_scores, dtype=np.float64)
    n = np.array(client_sizes, dtype=np.float64)
    Tn = T * n
    total = Tn.sum()
    
    if np.isnan(total) or total < EPSILON:
        alpha = n / n.sum()
    else:
        alpha = Tn / total
    
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = alpha / alpha.sum()
    
    aggregated = []
    for layer_idx in range(len(client_weights_list[0])):
        layer_agg = sum(
            alpha[c] * client_weights_list[c][layer_idx]
            for c in range(len(client_weights_list))
        )
        aggregated.append(layer_agg)
    
    return aggregated, alpha

# ===========================================================
# STEP 5: DTS TRAINING LOOP
# ===========================================================
print("\n" + "="*65)
print("STEP 5: DTS TRAINING LOOP")
print("="*65)

global_model = build_model(n_features)
global_weights = get_weights(global_model)
prev_losses = None

history = {
    'round': [], 'train_loss': [], 'train_acc': [],
    'val_acc': [], 'val_f1': [], 'global_loss': [],
    'trust_scores': [], 'alpha': [],
}

best_val_acc = 0.0
best_weights = global_weights
early_stop_cnt = 0
cnt_conv = 0
prev_global_loss = np.inf
actual_rounds = 0
stopped_by = "T_max"

print(f"\n{'Rd':>4} | {'GlobLoss':>9} | {'TrAcc':>7} | {'ValAcc':>7} | {'ValF1':>6}")
print("-"*50)

for t in range(1, T_MAX + 1):
    actual_rounds = t
    all_weights, all_sizes = [], []
    round_losses, round_accs = [], []
    
    for Xtr, ytr in client_trains:
        w, n, loss_k, acc_k = local_train(global_weights, Xtr, ytr, n_features)
        all_weights.append(w)
        all_sizes.append(n)
        round_losses.append(loss_k)
        round_accs.append(acc_k)
    
    trust_scores = compute_dts(round_accs, round_losses, prev_losses)
    global_weights, alpha = dts_aggregate(all_weights, all_sizes, trust_scores)
    set_weights(global_model, global_weights)
    
    global_loss_t = float(np.mean(round_losses))
    prev_losses = round_losses.copy()
    
    if abs(global_loss_t - prev_global_loss) < DELTA:
        cnt_conv += 1
    else:
        cnt_conv = 0
    prev_global_loss = global_loss_t
    
    if t % 5 == 0 or t == 1:
        val_accs, val_f1s = [], []
        for Xvl, yvl in client_vals:
            yhat = (global_model.predict(Xvl.astype(np.float32), verbose=0) > 0.5).astype(int).flatten()
            val_accs.append(accuracy_score(yvl, yhat))
            val_f1s.append(f1_score(yvl, yhat, zero_division=0))
        
        avg_val_acc = float(np.mean(val_accs))
        avg_val_f1 = float(np.mean(val_f1s))
        
        history['round'].append(t)
        history['train_loss'].append(float(np.mean(round_losses)))
        history['train_acc'].append(float(np.mean(round_accs)))
        history['val_acc'].append(avg_val_acc)
        history['val_f1'].append(avg_val_f1)
        history['global_loss'].append(global_loss_t)
        history['trust_scores'].append(trust_scores.tolist())
        history['alpha'].append(alpha.tolist())
        
        print(f"{t:>4} | {global_loss_t:>9.4f} | {np.mean(round_accs):>7.4f} | "
              f"{avg_val_acc:>7.4f} | {avg_val_f1:>6.4f}")
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_weights = [w.copy() for w in global_weights]
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        
        if early_stop_cnt >= PATIENCE:
            stopped_by = f"Early stopping (patience={PATIENCE})"
            break
    
    if cnt_conv >= TAU:
        stopped_by = f"Konvergensi loss global (τ={TAU} round)"
        break
    
    if t >= T_MAX:
        stopped_by = f"T_max={T_MAX} tercapai"
        break

set_weights(global_model, best_weights)
print(f"\nTraining berhenti: {stopped_by}")
print(f"Total rounds: {actual_rounds}")
print(f"Best val accuracy: {best_val_acc:.4f}")

# ===========================================================
# STEP 6: EVALUASI AKHIR
# ===========================================================
print("\n" + "="*65)
print("STEP 6: EVALUASI AKHIR — TEST SET GLOBAL")
print("="*65)

def evaluate_model(model, X, y_true, model_name):
    y_prob = model.predict(X.astype(np.float32), verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"\n  {model_name}")
    print(f"  Accuracy  : {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f} | AUC: {auc:.4f} | FPR: {fpr:.4f}")
    return {'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'auc': auc, 'fpr': fpr, 'y_pred': y_pred}

dtsa_res = evaluate_model(global_model, X_test_sc, y_test, "DTS")

# Baseline FedAvg
print("\nTraining Baseline FedAvg...")

def fedavg_aggregate_baseline(client_weights_list, client_sizes):
    total = sum(client_sizes)
    aggregated = []
    for layer_idx in range(len(client_weights_list[0])):
        layer_agg = sum(
            (client_weights_list[c][layer_idx] * client_sizes[c] / total)
            for c in range(len(client_weights_list))
        )
        aggregated.append(layer_agg)
    return aggregated

fedavg_model = build_model(n_features)
fedavg_weights = get_weights(fedavg_model)
fedavg_best = fedavg_weights
fedavg_best_acc = 0.0
fedavg_prev_losses = None
fedavg_cnt_conv = 0
fedavg_prev_global_loss = np.inf

for t in range(1, T_MAX + 1):
    all_w, all_n, all_l = [], [], []
    for Xtr, ytr in client_trains:
        w, n, loss_k, _ = local_train(fedavg_weights, Xtr, ytr, n_features)
        all_w.append(w)
        all_n.append(n)
        all_l.append(loss_k)

    fedavg_weights = fedavg_aggregate_baseline(all_w, all_n)
    set_weights(fedavg_model, fedavg_weights)

    gl = float(np.mean(all_l))
    if abs(gl - fedavg_prev_global_loss) < DELTA:
        fedavg_cnt_conv += 1
    else:
        fedavg_cnt_conv = 0
    fedavg_prev_global_loss = gl

    if t % 5 == 0 or t == 1:
        val_accs = []
        for Xvl, yvl in client_vals:
            yhat = (fedavg_model.predict(Xvl.astype(np.float32), verbose=0) > 0.5).astype(int).flatten()
            val_accs.append(accuracy_score(yvl, yhat))
        avg = float(np.mean(val_accs))
        if avg > fedavg_best_acc:
            fedavg_best_acc = avg
            fedavg_best = [w.copy() for w in fedavg_weights]

    if fedavg_cnt_conv >= TAU or t >= T_MAX:
        print(f"  FedAvg selesai di round {t}")
        break

set_weights(fedavg_model, fedavg_best)
fedavg_res = evaluate_model(fedavg_model, X_test_sc, y_test, "FedAvg (Baseline)")

# Baseline Centralized
print("\nTraining Centralized Baseline...")
cent_model = build_model(n_features)
cb_es = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=20,
    restore_best_weights=True, verbose=0
)
cent_model.fit(
    X_tv_sc.astype(np.float32), y_tv.astype(np.float32),
    validation_split=0.15, epochs=200, batch_size=BATCH_SIZE,
    callbacks=[cb_es], verbose=0
)
cent_res = evaluate_model(cent_model, X_test_sc, y_test, "Centralized")

# ===========================================================
# PERBANDINGAN
# ===========================================================
print("\n" + "="*65)
print("PERBANDINGAN HASIL")
print("="*65)
print(f"\n  {'Model':<20} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
print("  " + "-"*65)
for name, res in [("DTS", dtsa_res), ("FedAvg", fedavg_res), ("Centralized", cent_res)]:
    print(f"  {name:<20} {res['accuracy']:>7.4f} {res['precision']:>7.4f} "
          f"{res['recall']:>7.4f} {res['f1']:>7.4f} {res['auc']:>7.4f}")

print("\n✅ EKSPERIMEN SELESAI!")
