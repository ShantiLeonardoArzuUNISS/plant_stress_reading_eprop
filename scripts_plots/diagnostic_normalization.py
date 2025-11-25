"""
diagnostic_normalization.py

Script diagnostico per verificare la normalizzazione e identificare eventuali problemi.
Analizza i dati RAW e NORMALIZZATI per capire la causa dei valori estremi.

Author: Shanti Leonardo Arzu
Date: November 2025
"""

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data(stress_type):
    """Carica i dati dal file .npz"""
    file_path = f"../data/{stress_type.capitalize()}_Stress.npz"

    if not os.path.exists(file_path):
        print(f"❌ File non trovato: {file_path}")
        return None, None, None

    data = np.load(file_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    plant_ids = data["plant_ids"]

    return X, y, plant_ids


def analyze_raw_data(X):
    """Analizza i dati RAW per identificare anomalie"""
    X_shaped = X.reshape(-1, 200, 2)

    print("\n" + "="*80)
    print("📊 ANALISI DATI RAW")
    print("="*80)

    # Statistiche globali
    print("\n1. STATISTICHE GLOBALI:")
    print(f"   Shape: {X_shaped.shape}")
    print(f"   Real Part - Min: {X_shaped[:,:,0].min():.2f}, Max: {X_shaped[:,:,0].max():.2f}")
    print(f"   Imag Part - Min: {X_shaped[:,:,1].min():.2f}, Max: {X_shaped[:,:,1].max():.2f}")
    print(f"   Real Part - Mean: {X_shaped[:,:,0].mean():.2f}, Std: {X_shaped[:,:,0].std():.2f}")
    print(f"   Imag Part - Mean: {X_shaped[:,:,1].mean():.2f}, Std: {X_shaped[:,:,1].std():.2f}")

    # Analisi per frequenza
    print("\n2. ANALISI PER FREQUENZA (Real Part):")
    real_means = X_shaped[:,:,0].mean(axis=0)
    real_stds = X_shaped[:,:,0].std(axis=0)

    print(f"   Media per frequenza - Min: {real_means.min():.2f}, Max: {real_means.max():.2f}")
    print(f"   Std per frequenza - Min: {real_stds.min():.4f}, Max: {real_stds.max():.2f}")
    print(f"   Frequenze con std < 0.01: {np.sum(real_stds < 0.01)}/200")
    print(f"   Frequenze con std < 0.1: {np.sum(real_stds < 0.1)}/200")

    # Identifica frequenze problematiche
    low_std_freq = np.where(real_stds < 0.1)[0]
    if len(low_std_freq) > 0:
        print(f"\n   ⚠️  {len(low_std_freq)} frequenze hanno std molto bassa:")
        print(f"      Indici: {low_std_freq[:10]}..." if len(low_std_freq) > 10 else f"      Indici: {low_std_freq}")

    print("\n3. ANALISI PER FREQUENZA (Imaginary Part):")
    imag_means = X_shaped[:,:,1].mean(axis=0)
    imag_stds = X_shaped[:,:,1].std(axis=0)

    print(f"   Media per frequenza - Min: {imag_means.min():.2f}, Max: {imag_means.max():.2f}")
    print(f"   Std per frequenza - Min: {imag_stds.min():.4f}, Max: {imag_stds.max():.2f}")
    print(f"   Frequenze con std < 0.01: {np.sum(imag_stds < 0.01)}/200")
    print(f"   Frequenze con std < 0.1: {np.sum(imag_stds < 0.1)}/200")

    # Verifica outliers nei dati raw
    print("\n4. VERIFICA OUTLIERS (Real Part):")
    for sample_idx in range(min(5, len(X_shaped))):
        sample_min = X_shaped[sample_idx, :, 0].min()
        sample_max = X_shaped[sample_idx, :, 0].max()
        print(f"   Campione {sample_idx}: Min={sample_min:.2f}, Max={sample_max:.2f}")

    return X_shaped, real_stds, imag_stds


def analyze_normalized_data(X, y):
    """Analizza i dati NORMALIZZATI"""
    X_shaped = X.reshape(-1, 200, 2)

    # Split come nel codice originale
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_shaped, y, test_size=0.3, shuffle=True, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=True, stratify=y_temp, random_state=42
    )

    # Normalizzazione
    mean_train = np.mean(X_train, axis=0, keepdims=True)
    std_train = np.std(X_train, axis=0, keepdims=True)
    std_train[std_train == 0] = 1.0

    X_normalized = (X_shaped - mean_train) / std_train

    print("\n" + "="*80)
    print("📊 ANALISI DATI NORMALIZZATI")
    print("="*80)

    print("\n1. STATISTICHE NORMALIZZAZIONE:")
    print(f"   Train mean - Range: [{mean_train.min():.4f}, {mean_train.max():.4f}]")
    print(f"   Train std - Range: [{std_train.min():.4f}, {std_train.max():.4f}]")
    print(f"   Frequenze con std sostituita (==1.0): {np.sum(std_train == 1.0)}")

    print("\n2. STATISTICHE DATI NORMALIZZATI:")
    print(f"   Real Part - Min: {X_normalized[:,:,0].min():.2f}, Max: {X_normalized[:,:,0].max():.2f}")
    print(f"   Imag Part - Min: {X_normalized[:,:,1].min():.2f}, Max: {X_normalized[:,:,1].max():.2f}")
    print(f"   Real Part - Mean: {X_normalized[:,:,0].mean():.4f}, Std: {X_normalized[:,:,0].std():.4f}")
    print(f"   Imag Part - Mean: {X_normalized[:,:,1].mean():.4f}, Std: {X_normalized[:,:,1].std():.4f}")

    # Identifica campioni con valori estremi
    print("\n3. CAMPIONI CON VALORI ESTREMI (|z| > 5):")
    extreme_mask_real = np.abs(X_normalized[:,:,0]) > 5
    extreme_mask_imag = np.abs(X_normalized[:,:,1]) > 5

    samples_with_extremes_real = np.any(extreme_mask_real, axis=1)
    samples_with_extremes_imag = np.any(extreme_mask_imag, axis=1)

    print(f"   Campioni con valori estremi (Real): {np.sum(samples_with_extremes_real)}/{len(X_normalized)}")
    print(f"   Campioni con valori estremi (Imag): {np.sum(samples_with_extremes_imag)}/{len(X_normalized)}")

    if np.sum(samples_with_extremes_real) > 0:
        extreme_indices = np.where(samples_with_extremes_real)[0][:5]
        print(f"\n   Primi 5 indici con valori estremi (Real): {extreme_indices}")
        for idx in extreme_indices:
            max_val = X_normalized[idx, :, 0].max()
            min_val = X_normalized[idx, :, 0].min()
            freq_idx_max = X_normalized[idx, :, 0].argmax()
            freq_idx_min = X_normalized[idx, :, 0].argmin()
            print(f"      Sample {idx}: Max={max_val:.2f} (freq {freq_idx_max}), Min={min_val:.2f} (freq {freq_idx_min})")

    # Analisi per frequenza
    print("\n4. FREQUENZE CON VALORI ESTREMI:")
    freq_max_values = X_normalized[:,:,0].max(axis=0)
    freq_min_values = X_normalized[:,:,0].min(axis=0)

    extreme_freqs = np.where((freq_max_values > 5) | (freq_min_values < -5))[0]
    print(f"   Frequenze con valori |z| > 5: {len(extreme_freqs)}/200")
    if len(extreme_freqs) > 0:
        print(f"   Indici: {extreme_freqs}")
        for freq_idx in extreme_freqs[:5]:
            print(f"      Freq {freq_idx}: Range [{freq_min_values[freq_idx]:.2f}, {freq_max_values[freq_idx]:.2f}]")

    return X_normalized, mean_train, std_train


def main():
    print("\n" + "="*80)
    print("🔍 DIAGNOSTICA NORMALIZZAZIONE - ANALISI OUTLIERS")
    print("="*80)

    stress_type = "water"  # Cambia in "iron" se necessario

    print(f"\n⏳ Caricamento dataset {stress_type}...")
    X, y, plant_ids = load_data(stress_type)

    if X is None:
        return

    # Analisi dati RAW
    X_shaped, real_stds, imag_stds = analyze_raw_data(X)

    # Analisi dati NORMALIZZATI
    X_normalized, mean_train, std_train = analyze_normalized_data(X, y)

    # Conclusioni
    print("\n" + "="*80)
    print("💡 CONCLUSIONI")
    print("="*80)

    # Check se ci sono problemi con le std
    low_std_count = np.sum(real_stds < 0.1)
    if low_std_count > 50:
        print(f"\n⚠️  PROBLEMA POTENZIALE: {low_std_count} frequenze hanno std < 0.1")
        print("   Questo può causare valori normalizzati estremi anche con piccole variazioni.")
        print("   SOLUZIONE: Considerare clipping dei valori normalizzati o normalizzazione robusta.")

    extreme_count = np.sum(np.abs(X_normalized[:,:,0]) > 5)
    total_values = X_normalized[:,:,0].size
    extreme_percentage = (extreme_count / total_values) * 100

    print(f"\n📊 PERCENTUALE VALORI ESTREMI (|z| > 5):")
    print(f"   {extreme_count}/{total_values} = {extreme_percentage:.2f}%")

    if extreme_percentage > 1:
        print("   ⚠️  Alta percentuale di valori estremi - possibili outliers reali o problema acquisizione")
    elif extreme_percentage > 0.1:
        print("   ⚡ Presenza di outliers - normale per dataset reali")
    else:
        print("   ✅ Normalizzazione corretta - pochi outliers")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
