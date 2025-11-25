"""
inspect_npz_simple.py

Ispeziona il contenuto di Iron_Stress.npz o Water_Stress.npz

Utilizzo:
    python inspect_npz_simple.py iron
    python inspect_npz_simple.py water
"""

from pathlib import Path

import numpy as np

# Carica il file
stress_type = input("Quale stress? (iron/water): ").strip().lower()
file_path = Path(f"../data/{stress_type.capitalize()}_Stress.npz")

if not file_path.exists():
    print(f"File non trovato: {file_path}")
    exit()

print(f"\nCaricamento: {file_path}")
data = np.load(file_path, allow_pickle=True)

# Mostra array contenuti
print(f"\nArray nel file: {list(data.files)}")
print(f"Dimensione totale: {file_path.stat().st_size / (1024**2):.2f} MB\n")

# === FEATURES (X) ===
X = data["X"]
print("=" * 60)
print("FEATURES (X)")
print("=" * 60)
print(f"Shape: {X.shape}")
print(f"Tipo: {X.dtype}")
print(f"Min: {X.min():.6f} | Max: {X.max():.6f}")
print(f"Media: {X.mean():.6f} | Std: {X.std():.6f}")
print(f"Memoria: {X.nbytes / (1024**2):.2f} MB")

# Analisi formato
num_freq = X.shape[1] // 2
print(f"\nOrganizzazione: {num_freq} frequenze × 2 componenti (Real + Imaginary)")
print("Formato Flattened: [R₀, I₀, R₁, I₁, ..., R₁₉₉, I₁₉₉]")
print("\nIndici 3 frequenze ottimali:")
print("  Water Stress (100-120 Hz):    [0, 1, 2]")
print("  Iron Stress (4.7-10 MHz):     [197, 198, 199]")

# === LABELS (y) ===
y = data["y"]
print("\n" + "=" * 60)
print("LABELS (y)")
print("=" * 60)
print(f"Shape: {y.shape}")
print(f"Numero classi: {len(np.unique(y))}")

# Conta per classe
classes, counts = np.unique(y, return_counts=True)
class_names = {0: "Control", 1: "Early Stress", 2: "Late Stress"}

print("\nDistribuzione classi:")
for cls, count in zip(classes, counts):
    name = class_names.get(int(cls), f"Classe {cls}")
    pct = (count / len(y)) * 100
    print(f"  {name:15} {count:4d} ({pct:5.1f}%) ")

# === PLANT IDS ===
if "plant_ids" in data.files:
    plant_ids = data["plant_ids"]
    print("\n" + "=" * 60)
    print("PLANT IDS")
    print("=" * 60)

    plants = np.unique(plant_ids)
    print(f"Numero piante: {len(plants)}")
    print(f"IDs: {list(plants)}\n")

    for plant in sorted(plants):
        mask = plant_ids == plant
        count = np.sum(mask)
        pct = (count / len(plant_ids)) * 100
        print(f"  {str(plant):5} {count:4d} campioni ({pct:5.1f}%) ")

print("\n" + "=" * 60)
print("✓ Ispezione completata!")
print("=" * 60)
