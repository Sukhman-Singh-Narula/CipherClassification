import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

d = np.load("/Users/sukhmansinghnarula/.claude-science/orgs/d2396652-2759-4667-bd71-2839c2fe85f6/artifacts/proj_9d30ad28589c/275b81eb-841d-4011-aa07-4c866289c7c9/v6ceb3a12_keystreams_max1024.npz")
rc4_bytes, triv_bytes, esp_bytes = d['rc4'], d['trivium'], d['espresso']
N = rc4_bytes.shape[0]
print("loaded:", N, rc4_bytes.shape)

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
    cm = confusion_matrix(yte, preds).tolist()
    n_epochs_run = len(hist.history['loss'])
    return {"test_acc": float(test_acc), "test_loss": float(test_loss),
            "confusion_matrix": cm, "n_epochs": n_epochs_run, "train_time_s": train_time,
            "n_per_cipher": n_per_cipher}

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
with open("results.json","w") as f:
    json.dump({"results": results, "deviations": deviations}, f, indent=2)