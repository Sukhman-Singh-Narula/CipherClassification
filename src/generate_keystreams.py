import numpy as np
import time
import os

def rc4_keystream_batch(keys, n_bytes):
    n, keylen = keys.shape
    S = np.tile(np.arange(256, dtype=np.uint8), (n, 1))
    j = np.zeros(n, dtype=np.int64)
    idx = np.arange(n)
    for i in range(256):
        j = (j + S[:, i].astype(np.int64) + keys[:, i % keylen].astype(np.int64)) % 256
        Si = S[idx, i].copy()
        Sj = S[idx, j].copy()
        S[idx, i] = Sj
        S[idx, j] = Si
    out = np.empty((n, n_bytes), dtype=np.uint8)
    i_ = np.zeros(n, dtype=np.int64)
    j_ = np.zeros(n, dtype=np.int64)
    for t in range(n_bytes):
        i_ = (i_ + 1) % 256
        j_ = (j_ + S[idx, i_].astype(np.int64)) % 256
        Si = S[idx, i_].copy()
        Sj = S[idx, j_].copy()
        S[idx, i_] = Sj
        S[idx, j_] = Si
        out[:, t] = S[idx, (Si.astype(np.int64) + Sj.astype(np.int64)) % 256]
    return out

def bytes_to_bits(arr_bytes, n_bits, bitorder='little'):
    n = arr_bytes.shape[0]
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
        t1 = A[:, 65] ^ A[:, 92]
        t2 = B[:, 68] ^ B[:, 83]
        t3 = C[:, 65] ^ C[:, 110]
        z = None
        if keystream_output:
            z = t1 ^ t2 ^ t3
        t1n = t1 ^ (A[:, 90] & A[:, 91]) ^ B[:, 77]
        t2n = t2 ^ (B[:, 81] & B[:, 82]) ^ C[:, 86]
        t3n = t3 ^ (C[:, 108] & C[:, 109]) ^ A[:, 68]
        newA = np.empty_like(A); newA[:,0] = t3n; newA[:,1:] = A[:,:-1]
        newB = np.empty_like(B); newB[:,0] = t1n; newB[:,1:] = B[:,:-1]
        newC = np.empty_like(C); newC[:,0] = t2n; newC[:,1:] = C[:,:-1]
        return newA, newB, newC, z

    for _ in range(4*288):
        A, B, C, _ = clock(A, B, C, keystream_output=False)

    out_bits = np.empty((n, n_bits), dtype=np.uint8)
    for t in range(n_bits):
        A, B, C, z = clock(A, B, C, keystream_output=True)
        out_bits[:, t] = z
    return out_bits

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

N = 131072  # 2^17
max_bytes = 1024
rng = np.random.default_rng(42)

os.makedirs("data", exist_ok=True)
t0=time.time()

# RC4
keys = rng.integers(0,256,size=(N,16),dtype=np.uint8)
rc4_bytes = rc4_keystream_batch(keys, max_bytes)

# TRIVIUM
tkeys = rng.integers(0,256,size=(N,10),dtype=np.uint8)
tivs  = rng.integers(0,256,size=(N,10),dtype=np.uint8)
triv_bits = trivium_keystream_batch(tkeys, tivs, max_bytes*8)
triv_bytes = np.packbits(triv_bits.reshape(N, max_bytes, 8), axis=2, bitorder='little').reshape(N, max_bytes)

# ESPRESSO-like
ekeys = rng.integers(0,256,size=(N,16),dtype=np.uint8)
eivs  = rng.integers(0,256,size=(N,16),dtype=np.uint8)
esp_bits = espresso_like_keystream_batch(ekeys, eivs, max_bytes*8)
esp_bytes = np.packbits(esp_bits.reshape(N, max_bytes, 8), axis=2, bitorder='little').reshape(N, max_bytes)

print("generation time:", time.time()-t0, "s")
print(rc4_bytes.shape, triv_bytes.shape, esp_bytes.shape)

np.savez_compressed("data/keystreams_max1024.npz",
                     rc4=rc4_bytes, trivium=triv_bytes, espresso=esp_bytes)
print("saved", os.path.getsize("data/keystreams_max1024.npz")/1e6, "MB")