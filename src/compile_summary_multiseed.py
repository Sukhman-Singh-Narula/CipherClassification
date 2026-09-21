import numpy as np
import json
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ── cipher implementations ──────────────────────────────────────────────────

def rc4_keystream_batch(keys, n_bytes):
    n, keylen = keys.shape
    S = np.tile(np.arange(256, dtype=np.uint8), (n, 1))
    j = np.zeros(n, dtype=np.int64)
    idx = np.arange(n)
    for i in range(256):
        j = (j + S[:, i].astype(np.int64) + keys[:, i % keylen].astype(np.int64)) % 256
        Si = S[idx, i].copy(); Sj = S[idx, j].copy()
        S[idx, i] = Sj; S[idx, j] = Si
    out = np.empty((n, n_bytes), dtype=np.uint8)
    i_ = np.zeros(n, dtype=np.int64); j_ = np.zeros(n, dtype=np.int64)
    for t in range(n_bytes):
        i_ = (i_ + 1) % 256
        j_ = (j_ + S[idx, i_].astype(np.int64)) % 256
        Si = S[idx, i_].copy(); Sj = S[idx, j_].copy()
        S[idx, i_] = Sj; S[idx, j_] = Si
        out[:, t] = S[idx, (Si.astype(np.int64) + Sj.astype(np.int64)) % 256]
    return out

def bytes_to_bits(arr_bytes, n_bits, bitorder='little'):
    bits = np.unpackbits(arr_bytes, axis=1, bitorder='little')
    return bits[:, :n_bits]

def trivium_keystream_batch(keys80, ivs80, n_bits):
    n = keys80.shape[0]
    key_bits = bytes_to_bits(keys80, 80)
    iv_bits = bytes_to_bits(ivs80, 80)
    A = np.zeros((n, 93), dtype=np.uint8)
    B = np.zeros((n, 84), dtype=np.uint8)
    C = np.zeros((n, 111), dtype=np.uint8)
    A[:, 0:80] = key_bits
    B[:, 0:80] = iv_bits
    C[:, 108:111] = 1

    def clock(A, B, C, keystream_output=False):
        t1 = A[:, 65] ^ A[:, 92]; t2 = B[:, 68] ^ B[:, 83]; t3 = C[:, 65] ^ C[:, 110]
        z = (t1 ^ t2 ^ t3) if keystream_output else None
        t1n = t1 ^ (A[:, 90] & A[:, 91]) ^ B[:, 77]
        t2n = t2 ^ (B[:, 81] & B[:, 82]) ^ C[:, 86]
        t3n = t3 ^ (C[:, 108] & C[:, 109]) ^ A[:, 68]
        newA = np.empty_like(A); newA[:,0]=t3n; newA[:,1:]=A[:,:-1]
        newB = np.empty_like(B); newB[:,0]=t1n; newB[:,1:]=B[:,:-1]
        newC = np.empty_like(C); newC[:,0]=t2n; newC[:,1:]=C[:,:-1]
        return newA, newB, newC, z

    for _ in range(4*288):
        A,B,C,_ = clock(A,B,C,False)
    out = np.empty((n,n_bits), dtype=np.uint8)
    for t in range(n_bits):
        A,B,C,z = clock(A,B,C,True)
        out[:,t]=z
    return out

def espresso_like_keystream_batch(keys, ivs, n_bits, key_bits=128, iv_bits=128, state_len=256):
    n = keys.shape[0]
    key_b = np.unpackbits(keys, axis=1, bitorder='little')[:, :key_bits]
    iv_b = np.unpackbits(ivs, axis=1, bitorder='little')[:, :iv_bits]

    state = np.zeros((n, state_len), dtype=np.uint8)
    state[:, :key_bits] = key_b
    state[:, key_bits:key_bits+iv_bits] = iv_b
    state[:, -1] = 1

    taps = [255, 250, 192, 129, 90, 5]

    def clock(state, output=False):
        fb = state[:, taps[0]].copy()
        for t in taps[1:]:
            fb ^= state[:, t]
        z = None
        if output:
            z = (state[:, 80] ^ state[:, 99] ^ state[:, 137] ^ state[:, 227]
                 ^ state[:, 222] ^ state[:, 187] ^ (state[:, 243] & state[:, 217]))
        new_state = np.empty_like(state)
        new_state[:, 0] = fb
        new_state[:, 1:] = state[:, :-1]
        return new_state, z

    for _ in range(4 * state_len):
        state, _ = clock(state, output=False)

    out_bits = np.empty((n, n_bits), dtype=np.uint8)
    for t in range(n_bits):
        state, z = clock(state, output=True)
        out_bits[:, t] = z
    return out_bits

# ── model builders ───────────────────────────────────────────────────────────

def build_ann(L):
    m = models.Sequential([
        layers.Input(shape=(L,)),
        layers.Dense(4*L, activation='relu'),
        layers.Dense(8*L, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(3, activation='softmax'),
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def build_dnn(L):
    m = models.Sequential([
        layers.Input(shape=(L,)),
        layers.Dense(256, activation='relu'), layers.Dropout(0.3),
        layers.Dense(128, activation='relu'), layers.Dropout(0.3),
        layers.Dense(64, activation='relu'), layers.Dropout(0.3),
        layers.Dense(32, activation='relu'), layers.Dropout(0.3),
        layers.Dense(3, activation='softmax'),
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def build_cnn(L):
    m = models.Sequential([
        layers.Input(shape=(L,), dtype='int32'),
        layers.CategoryEncoding(num_tokens=256, output_mode='one_hot'),
        layers.Reshape((L, 256)),
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=2, padding='same'),
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=2, padding='same'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax'),
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def build_ffnn(n_features):
    m = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation='relu'), layers.Dropout(0.3),
        layers.Dense(32, activation='relu'), layers.Dropout(0.3),
        layers.Dense(3, activation='softmax'),
    ])
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

# ── feature extractor ────────────────────────────────────────────────────────

def extract_features(byte_arr):
    n, L = byte_arr.shape
    x = byte_arr.astype(np.float64)
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    centered = x - mean[:, None]
    std_safe = np.where(std == 0, 1, std)
    skew = (centered**3).mean(axis=1) / (std_safe**3)
    kurt = (centered**4).mean(axis=1) / (std_safe**4) - 3

    entropy = np.empty(n)
    hist_bins = 16
    hist = np.zeros((n, hist_bins))
    bin_idx = (byte_arr // (256 // hist_bins)).astype(np.int64)
    for b in range(hist_bins):
        hist[:, b] = (bin_idx == b).sum(axis=1)
    hist_prob = hist / L

    for i in range(n):
        counts = np.bincount(byte_arr[i], minlength=256).astype(np.float64)
        p = counts[counts > 0] / L
        entropy[i] = -(p * np.log2(p)).sum()

    mean_run = np.empty(n)
    max_run = np.empty(n)
    for i in range(n):
        row = byte_arr[i]
        changes = np.where(np.diff(row) != 0)[0]
        run_starts = np.concatenate(([0], changes + 1))
        run_ends = np.concatenate((changes + 1, [L]))
        runs = run_ends - run_starts
        mean_run[i] = runs.mean()
        max_run[i] = runs.max()

    feats = np.column_stack([mean, std, skew, kurt, entropy, mean_run, max_run, hist_prob])
    return feats.astype(np.float32)

# ── generate keystream data ──────────────────────────────────────────────────

N = 131072  # 2^17
max_bytes = 1024
rng = np.random.default_rng(42)

os.makedirs("data", exist_ok=True)

keys = rng.integers(0,256,size=(N,16),dtype=np.uint8)
rc4_bytes = rc4_keystream_batch(keys, max_bytes)

tkeys = rng.integers(0,256,size=(N,10),dtype=np.uint8)
tivs  = rng.integers(0,256,size=(N,10),dtype=np.uint8)
triv_bits = trivium_keystream_batch(tkeys, tivs, max_bytes*8)
triv_bytes = np.packbits(triv_bits.reshape(N, max_bytes, 8), axis=2, bitorder='little').reshape(N, max_bytes)

ekeys = rng.integers(0,256,size=(N,16),dtype=np.uint8)
eivs  = rng.integers(0,256,size=(N,16),dtype=np.uint8)
esp_bits = espresso_like_keystream_batch(ekeys, eivs, max_bytes*8)
esp_bytes = np.packbits(esp_bits.reshape(N, max_bytes, 8), axis=2, bitorder='little').reshape(N, max_bytes)

np.savez_compressed("data/keystreams_max1024.npz",
                     rc4=rc4_bytes, trivium=triv_bytes, espresso=esp_bytes)

# ── training harness ─────────────────────────────────────────────────────────

def get_data_for_length(L, n_per_cipher=None):
    if n_per_cipher is None:
        n_per_cipher = N
    idx = np.arange(n_per_cipher)
    X = np.concatenate([rc4_bytes[idx, :L], triv_bytes[idx, :L], esp_bytes[idx, :L]], axis=0)
    y = np.concatenate([np.zeros(n_per_cipher), np.ones(n_per_cipher), np.full(n_per_cipher, 2)]).astype(np.int64)
    return X, y

def run_one(model_name, L, n_per_cipher, max_epochs=20, patience=5, seed=0):
    X, y = get_data_for_length(L, n_per_cipher)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    ytr_oh = tf.keras.utils.to_categorical(ytr, 3)
    yte_oh = tf.keras.utils.to_categorical(yte, 3)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    if model_name == 'ANN':
        m = build_ann(L)
        Xtr_in, Xte_in = Xtr.astype(np.float32)/255.0, Xte.astype(np.float32)/255.0
    elif model_name == 'DNN':
        m = build_dnn(L)
        Xtr_in, Xte_in = Xtr.astype(np.float32)/255.0, Xte.astype(np.float32)/255.0
    elif model_name == '1D-CNN':
        m = build_cnn(L)
        Xtr_in, Xte_in = Xtr.astype(np.int32), Xte.astype(np.int32)
    elif model_name == 'FFNN':
        Xtr_f = extract_features(Xtr); Xte_f = extract_features(Xte)
        m = build_ffnn(Xtr_f.shape[1])
        Xtr_in, Xte_in = Xtr_f, Xte_f

    t0 = time.time()
    hist = m.fit(Xtr_in, ytr_oh, validation_split=0.1, epochs=max_epochs, batch_size=64,
                 verbose=0, callbacks=[es])
    train_time = time.time() - t0
    test_loss, test_acc = m.evaluate(Xte_in, yte_oh, verbose=0)
    preds = np.argmax(m.predict(Xte_in, verbose=0), axis=1)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yte, preds).tolist()
    n_epochs_run = len(hist.history['loss'])
    return {"test_acc": float(test_acc), "test_loss": float(test_loss),
            "confusion_matrix": cm, "n_epochs": n_epochs_run, "train_time_s": train_time,
            "n_per_cipher": n_per_cipher}

# ── seed=0 grid ──────────────────────────────────────────────────────────────

lengths = [8, 24, 32, 64, 1024]
models_list = ['ANN', 'DNN', '1D-CNN', 'FFNN']
results = {}
deviations = []
log = []

t_start = time.time()
for L in lengths:
    for mname in models_list:
        if L == 1024 and mname in ('ANN', '1D-CNN'):
            n_pc = 12000
            deviations.append(f"{mname} at L=1024: used {n_pc} samples/cipher instead of full {N} "
                               f"(reduced from full 2^17 due to CPU compute-time cost of large dense/one-hot layers at this length)")
        else:
            n_pc = N
        t0 = time.time()
        res = run_one(mname, L, n_pc, max_epochs=20, patience=5)
        dt = time.time() - t0
        results[f"{mname}_{L}"] = res
        log.append(f"{mname} L={L}: acc={res['test_acc']:.4f} epochs={res['n_epochs']} time={dt:.1f}s n_pc={n_pc}")
        print(log[-1])

print("TOTAL TIME:", time.time()-t_start, "s")
with open("data/results.json","w") as f:
    json.dump({"results": results, "deviations": deviations}, f, indent=2)

# ── extra seeds grid ─────────────────────────────────────────────────────────

extra_results = {}
t_start = time.time()
for L in lengths:
    for mname in models_list:
        n_pc = 12000 if (L == 1024 and mname in ('ANN','1D-CNN')) else N
        for seed in (1, 2):
            t0 = time.time()
            res = run_one(mname, L, n_pc, max_epochs=20, patience=5, seed=seed)
            dt = time.time() - t0
            extra_results[f"{mname}_{L}_seed{seed}"] = res
            print(f"{mname} L={L} seed={seed}: acc={res['test_acc']:.4f} time={dt:.1f}s")

print("TOTAL EXTRA SEED TIME:", time.time()-t_start, "s")
with open("data/extra_seeds.json","w") as f:
    json.dump(extra_results, f, indent=2)

# ── load and compute summary ─────────────────────────────────────────────────

with open("data/results.json") as f:
    saved = json.load(f)
results_seed0 = saved["results"]
deviations = saved["deviations"]

with open("data/extra_seeds.json") as f:
    extra = json.load(f)

lengths = [8, 24, 32, 64, 1024]
models_list = ['ANN', 'DNN', '1D-CNN', 'FFNN']

summary = {}
for L in lengths:
    for mname in models_list:
        accs = [results_seed0[f"{mname}_{L}"]["test_acc"]]
        for seed in (1,2):
            accs.append(extra[f"{mname}_{L}_seed{seed}"]["test_acc"])
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=1))
        cm0 = results_seed0[f"{mname}_{L}"]["confusion_matrix"]
        collapsed = any(all(cm0[r][c]==0 for r in range(3) if r!=c) is False for c in range(3))  # not used directly
        # detect collapse: some column has near-all the mass, or a class row is entirely predicted as one column
        cm_arr = np.array(cm0)
        col_sums = cm_arr.sum(axis=0)
        collapse_frac = col_sums.max() / col_sums.sum()
        summary[f"{mname}_{L}"] = {"seed_accs": accs, "mean_acc": mean_acc, "std_acc": std_acc,
                                     "collapse_frac_to_dominant_class": float(collapse_frac)}
        print(f"{mname:8s} L={L:5d}  mean={mean_acc:.4f} std={std_acc:.4f}  seeds={[round(a,4) for a in accs]}  collapse_frac={collapse_frac:.3f}")

with open("data/summary_multiseed.json","w") as f:
    json.dump(summary, f, indent=2)