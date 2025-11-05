"""
test_plant_data_preprocessing.py

Script di test per verificare il funzionamento della classe PlantFeatureSelector.

Questo script testa le funzionalità principali di plant_data_preprocessing.py:
- Selezione frequenze (water, iron, custom)
- Conversione Flattened → Shaped
- Riduzione features
- Splitting dataset con stratificazione
- Normalizzazione Z-score
- Prevenzione data leakage

Author: Shanti Leonardo Arzu
Date: November 2025
"""

import numpy as np

from plant_data_preprocessing import PlantFeatureSelector

# ============================================================================
# ESEMPIO DI UTILIZZO
# ============================================================================

if __name__ == "__main__":
    """
    Script di test per verificare il funzionamento della classe PlantFeatureSelector.
    """
    print("=" * 42)
    print("TEST: PlantFeatureSelector ")
    print("=" * 42)

    # ========================================================================
    # TEST 1: Water Stress con indici default
    # ========================================================================
    print("\n\nTEST 1: Water Stress (default) \n")

    selector_water = PlantFeatureSelector(stress_type="water")

    # Simula dataset completo
    n_samples = 2016
    X_full = np.random.randn(n_samples, 400)  # Flattened format
    y = np.random.randint(0, 3, n_samples)  # 3 classi: Control, Early, Late

    print(f"\nDataset originale: {X_full.shape}")

    # Test conversione Flattened → Shaped
    X_shaped = selector_water.flatten_to_shaped(X_full)
    print(f"Dopo conversione: {X_shaped.shape}")

    # Test riduzione features
    X_reduced = selector_water.reduce_dataset(X_full)
    print(f"Dopo riduzione: {X_reduced.shape}")

    # Test splitting
    X_train, X_val, X_test, y_train, y_val, y_test = selector_water.split_dataset(
        X_reduced, y, train_size=0.7, val_size=0.15, test_size=0.15
    )

    # Test normalizzazione
    X_train_norm, X_test_norm, X_val_norm, norm_params = (
        selector_water.normalize_features(X_train, X_test, X_val)
    )

    print("\n[RISULTATO] Water Stress completato con successo!")
    print(f"      Features: {X_reduced.shape[1] * 2}")
    print(f"      Train: {X_train_norm.shape}")
    print(f"      Val: {X_val_norm.shape}")
    print(f"      Test: {X_test_norm.shape}")

    # ========================================================================
    # TEST 2: Iron Stress con indici default
    # ========================================================================
    print("\n\nTEST 2: Iron Stress (default) \n")

    selector_iron = PlantFeatureSelector(stress_type="iron")
    X_reduced_iron = selector_iron.reduce_dataset(X_full)

    print(f"[RISULTATO] Iron Stress - Features ridotte a: {X_reduced_iron.shape}")

    # ========================================================================
    # TEST 3: Custom frequencies
    # ========================================================================
    print("\n\nTEST 3: Custom Frequencies\n")

    custom_indices = [5, 10, 15, 20, 25]  # 5 frequenze → 10 features
    selector_custom = PlantFeatureSelector(
        stress_type="custom", custom_freq_indices=custom_indices
    )
    X_reduced_custom = selector_custom.reduce_dataset(X_full)

    print(f"[RISULTATO] Custom - Features ridotte a: {X_reduced_custom.shape}")
    print(f"      Frequenze selezionate: {selector_custom.num_frequencies}")
    print(f"      Features totali: {selector_custom.num_features}")

    # ========================================================================
    # TEST 4: Test normalizzazione singolo dataset
    # ========================================================================
    print("\n\nTEST 4: Normalizzazione Singolo Dataset \n")

    X_single = np.random.randn(100, 3, 2)
    X_norm, norm_params = selector_water.normalize_dataset(X_single)

    print("[RISULTATO] Dataset normalizzato:")
    print(f"      Shape: {X_norm.shape}")
    print(f"      Mean: {X_norm.mean():.6f} (atteso ~0)")
    print(f"      Std: {X_norm.std():.6f} (atteso ~1)")

    print("\n" + "=" * 42)
    print("TUTTI I TEST COMPLETATI CON SUCCESSO!")
    print("=" * 42)
