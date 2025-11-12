"""
test_pipeline.py
Test end-to-end della pipeline di preprocessing
"""

import os
import sys

import numpy as np

# ==============================================================
# FIX PATH: Ottieni la directory dello script
# ==============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Parent della cartella scripts

# Aggiungi project_root al sys.path per importare i moduli
sys.path.insert(0, project_root)

# Path assoluto al dataset
DATA_PATH = os.path.join(project_root, "data", "Water_Stress.npz")

print(f"\n[DEBUG] Script directory: {script_dir}")
print(f"[DEBUG] Project root: {project_root}")
print(f"[DEBUG] Data path: {DATA_PATH}")
print(f"[DEBUG] File exists: {os.path.exists(DATA_PATH)}")

from plant_data_management import PlantDataManager


def test_standard_split():
    """Test split standard 70/15/15"""
    print("\n" + "=" * 70)
    print("TEST 1: STANDARD SPLIT (70/15/15)")
    print("=" * 70)

    # Inizializza data manager
    data_manager = PlantDataManager(
        stress_type="water",
        encoding_params={
            "encoding_type": "rate",
            "nb_steps": 100,
            "dt": 1.0,
            "gain": 10.0,
        },
    )

    # Prepara dataset - USA IL PATH CORRETTO
    ds_train, ds_val, ds_test, metadata = data_manager.prepare_dataset_standard_split(
        file_path=DATA_PATH,  # ← CAMBIATO QUI
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
    )

    # Verifica dimensioni
    print("\nVERIFICA DATASET:")
    print(f"   Train: {len(ds_train)} samples")
    print(f"   Val: {len(ds_val)} samples")
    print(f"   Test: {len(ds_test)} samples")

    # Verifica shape di un batch
    X_sample, y_sample = ds_train[0]
    print("\nVERIFICA SHAPE:")
    print(f"   Input shape: {X_sample.shape}")  # Dovrebbe essere (100, 6)
    print(f"   Label shape: {y_sample.shape}")  # Dovrebbe essere scalare
    print(f"   Label value: {y_sample.item()}")  # 0, 1 o 2

    # Verifica spike activity
    spike_rate = (X_sample.sum() / X_sample.numel()) * 100
    print("\nVERIFICA SPIKE ACTIVITY:")
    print(f"   Spike rate: {spike_rate:.2f}%")
    print("   Range consigliato: 20-50%")

    if spike_rate < 10 or spike_rate > 70:
        print("   ATTENZIONE: spike rate fuori range ottimale!")
        print("   Considera di modificare il parametro 'gain'")

    # Stampa metadata
    print("\nMETADATA:")
    for key, value in metadata.items():
        if key != "normalization_params":  # Troppo verbose
            print(f"   {key}: {value}")

    return ds_train, ds_val, ds_test, metadata


def test_leave_one_plant_split():
    """Test leave-one-plant-out split"""
    print("\n" + "=" * 70)
    print("TEST 2: LEAVE-ONE-PLANT-OUT (Train: P0+P1, Test: P3)")
    print("=" * 70)

    # Inizializza data manager
    data_manager = PlantDataManager(
        stress_type="water",
        encoding_params={
            "encoding_type": "rate",
            "nb_steps": 100,
            "dt": 1.0,
            "gain": 10.0,
        },
    )

    # Prepara dataset con leave-one-plant-out - USA IL PATH CORRETTO
    ds_train, ds_val, ds_test, metadata = (
        data_manager.prepare_dataset_leave_one_plant_split(
            file_path=DATA_PATH,  # ← CAMBIATO QUI
            leave_plant="P3",
            val_size=0.5,
            test_size=0.5,
        )
    )

    # Verifica dimensioni
    print("\nVERIFICA DATASET:")
    print(f"   Train (P0+P1): {len(ds_train)} samples")
    print(f"   Val (P3): {len(ds_val)} samples")
    print(f"   Test (P3): {len(ds_test)} samples")

    # Stampa metadata specifici LOPO
    print("\nLOPO METADATA:")
    print(f"   Strategy: {metadata['split_strategy']}")
    print(f"   Leave plant: {metadata['leave_plant']}")
    print(f"   Train plants: {metadata['train_plants']}")

    return ds_train, ds_val, ds_test, metadata


def test_encoding_methods():
    """Confronta diversi metodi di encoding"""
    print("\n" + "=" * 70)
    print("TEST 3: CONFRONTO METODI DI ENCODING")
    print("=" * 70)

    from plant_temporal_encoding import TemporalEncoder

    # Genera dati normalizzati di esempio
    X_sample = np.random.randn(10, 6)  # 10 campioni, 6 features

    encodings = ["rate", "latency", "temporal"]

    for enc_type in encodings:
        encoder = TemporalEncoder(encoding_type=enc_type, nb_steps=100, dt=1.0)
        spikes = encoder.encode(X_sample)

        spike_rate = (spikes.sum() / spikes.numel()) * 100

        print(f"\n{enc_type.upper()} Encoding:")
        print(f"   Output shape: {spikes.shape}")
        print(f"   Spike rate: {spike_rate:.2f}%")
        print(f"   Total spikes: {spikes.sum().item():.0f}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PLANT STRESS CLASSIFICATION - PIPELINE TEST")
    print("=" * 70)

    # Test 1: Standard split
    try:
        ds_train, ds_val, ds_test, metadata = test_standard_split()
        print("\nTest 1 PASSED: Standard split funziona correttamente")
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        import traceback

        traceback.print_exc()

    # Test 2: Leave-one-plant-out
    try:
        ds_train_lopo, ds_val_lopo, ds_test_lopo, metadata_lopo = (
            test_leave_one_plant_split()
        )
        print("\nTest 2 PASSED: LOPO split funziona correttamente")
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        import traceback

        traceback.print_exc()

    # Test 3: Encoding methods
    try:
        test_encoding_methods()
        print("\nTest 3 PASSED: Tutti i metodi di encoding funzionano")
    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("TUTTI I TEST COMPLETATI")
    print("=" * 70)
