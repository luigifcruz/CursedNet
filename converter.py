import numpy as np

raw_if = "../../PyRadio/IF_96900000.c64"   # Complex 64 @ 256e3
raw_fm = "../../PyRadio/FM_96900000.if32"  # Float 32 @ 32e3

raw_if = np.fromfile(raw_if, dtype=np.complex64)
raw_fm = np.fromfile(raw_fm, dtype=np.float32)

cif = int(256e3)
cfm = int(32e3)

for i in range(len(raw_if)//cif):
    if i > 0:
        np.save(f"dataset/IF_{i}.npy", raw_if[((i-1)*cif):(i*cif)])
        np.save(f"dataset/FM_{i}.npy", raw_fm[((i-1)*cfm):(i*cfm)])
        print(f"Saved {i}...")
