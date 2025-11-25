"""
plant_data_management.py
Data loading, preprocessing, and splitting for plant stress datasets

Author: Shanti Leonardo Arzu
Date: November 2025

"""

import numpy as np
import torch
from plant_data_preprocessing import PlantFeatureSelector
from plant_temporal_encoding import TemporalEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


class PlantDataManager:
    """
    Gestisce caricamento, preprocessing e splitting dei dataset piante.
    """

    def __init__(self, stress_type="water", encoding_params=None):
        """
        Args:
            stress_type (str): 'water' o 'iron'
            encoding_params (dict): Parametri per temporal encoding
        """
        self.stress_type = stress_type
        self.feature_selector = PlantFeatureSelector(stress_type)

        # Default encoding parameters
        if encoding_params is None:
            encoding_params = {
                "encoding_type": "rate",
                "nb_steps": 100,
                "dt": 1.0,
                "gain": 10.0,
            }

        self.temporal_encoder = TemporalEncoder(**encoding_params)

    def load_npz_data(self, file_path):
        """
        Carica dati da file .npz

        Returns:
            X (np.ndarray): Features (n_samples, 400)
            y (np.ndarray): Labels (n_samples,)
            plant_ids (np.ndarray): Plant IDs (n_samples,)
        """
        data = np.load(file_path, allow_pickle=True)

        X = data["X"]
        y = data["y"]
        plant_ids = data["plant_ids"]

        print(f"Loaded data from {file_path}")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Unique labels: {np.unique(y)}")
        print(f"  Unique plants: {np.unique(plant_ids)}")

        return X, y, plant_ids

    def prepare_dataset_standard_split(
        self, file_path, train_size=0.7, val_size=0.15, test_size=0.15
    ):
        """
        Prepara dataset con split standard (random)

        Returns:
            ds_train, ds_val, ds_test: TensorDatasets PyTorch
            metadata (dict): Informazioni sul dataset
        """
        # Step 1: Carica dati
        X, y, plant_ids = self.load_npz_data(file_path)

        # Step 2: Feature selection
        print("\n[1/5] Feature Selection...")
        X_selected = self.feature_selector.select_features(X)
        print(f"  Features reduced: {X.shape} â†’ {X_selected.shape}")

        # Step 3: Split dataset
        print("\n[2/5] Splitting dataset...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_selected, y, test_size=(val_size + test_size), stratify=y, random_state=42
        )

        val_fraction = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1 - val_fraction),
            stratify=y_temp,
            random_state=42,
        )

        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val:   {X_val.shape[0]} samples")
        print(f"  Test:  {X_test.shape[0]} samples")

        # Step 4: Normalizzazione (CORRETTO!)
        print("\n[3/5] Normalizing features...")
        X_train_norm, X_val_norm, X_test_norm, norm_params = (
            self.feature_selector.normalize_features(X_train, X_val, X_test)
        )

        # Step 5: Temporal encoding
        print("\n[4/5] Temporal encoding...")

        # Flatten per encoding: (n, nb_freqs, 2) â†’ (n, nb_freqs*2)
        X_train_flat = X_train_norm.reshape(X_train_norm.shape[0], -1)
        X_val_flat = X_val_norm.reshape(X_val_norm.shape[0], -1)
        X_test_flat = X_test_norm.reshape(X_test_norm.shape[0], -1)

        X_train_temporal = self.temporal_encoder.encode(X_train_flat)
        X_val_temporal = self.temporal_encoder.encode(X_val_flat)
        X_test_temporal = self.temporal_encoder.encode(X_test_flat)

        print(f"  Temporal shape: {X_train_temporal.shape}")

        # Step 6: Creazione TensorDatasets
        print("\n[5/5] Creating PyTorch datasets...")
        ds_train = TensorDataset(
            X_train_temporal, torch.tensor(y_train, dtype=torch.long)
        )
        ds_val = TensorDataset(X_val_temporal, torch.tensor(y_val, dtype=torch.long))
        ds_test = TensorDataset(X_test_temporal, torch.tensor(y_test, dtype=torch.long))

        # Metadata
        metadata = {
            "nb_inputs": self.feature_selector.nb_features,
            "nb_outputs": 3,
            "nb_steps": self.temporal_encoder.nb_steps,
            "dt": self.temporal_encoder.dt,
            "stress_type": self.stress_type,
            "normalization_params": norm_params,
            "selected_frequencies": self.feature_selector.selected_freq_indices,
            "split_strategy": "standard",
            "split_sizes": {"train": train_size, "val": val_size, "test": test_size},
        }

        print("\nâœ“ Standard split dataset preparation complete!")

        return ds_train, ds_val, ds_test, metadata

    def prepare_dataset_leave_one_plant_split(
        self, file_path, leave_plant, val_size=0.5, test_size=0.5
    ):
        """
        Prepara dataset con Leave-One-Plant-Out

        Args:
            file_path (str): Path al file .npz
            leave_plant (str): Pianta da escludere (es. 'P3')
            val_size (float): Frazione della pianta esclusa per validation
            test_size (float): Frazione della pianta esclusa per test

        Returns:
            ds_train, ds_val, ds_test: TensorDatasets PyTorch
            metadata (dict): Informazioni sul dataset
        """
        # Step 1: Carica dati
        X, y, plant_ids = self.load_npz_data(file_path)

        # Step 2: Feature selection
        print("\n[1/6] Feature Selection...")
        X_selected = self.feature_selector.select_features(X)
        print(f"  Features reduced: {X.shape} â†’ {X_selected.shape}")

        # Step 3: Split per pianta
        print(f"\n[2/6] Splitting by plant (leaving out {leave_plant})...")

        # Maschere per train (altre piante) e test (pianta esclusa)
        train_mask = plant_ids != leave_plant
        test_mask = plant_ids == leave_plant

        X_train = X_selected[train_mask]
        y_train = y[train_mask]
        train_plants = plant_ids[train_mask]

        X_test_full = X_selected[test_mask]
        y_test_full = y[test_mask]

        print(f"  Train plants: {np.unique(train_plants)} - {X_train.shape[0]} samples")
        print(f"  Test plant: {leave_plant} - {X_test_full.shape[0]} samples")

        # Step 4: Split validation e test dalla pianta esclusa
        print(f"\n[3/6] Splitting {leave_plant} into val/test...")

        if val_size + test_size != 1.0:
            print(f"  WARNING: val_size ({val_size}) + test_size ({test_size}) != 1.0")
            print("           Rinormalizzando...")
            total = val_size + test_size
            val_size = val_size / total
            test_size = test_size / total

        X_val, X_test, y_val, y_test = train_test_split(
            X_test_full,
            y_test_full,
            test_size=test_size,
            stratify=y_test_full,
            random_state=42,
        )

        print(f"  Val: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")

        # Step 5: Normalizzazione
        print("\n[4/6] Normalizing features...")
        X_train_norm, X_val_norm, X_test_norm, norm_params = (
            self.feature_selector.normalize_features(X_train, X_val, X_test)
        )

        # Step 6: Temporal encoding
        print("\n[5/6] Temporal encoding...")

        # Flatten per encoding
        X_train_flat = X_train_norm.reshape(X_train_norm.shape[0], -1)
        X_val_flat = X_val_norm.reshape(X_val_norm.shape[0], -1)
        X_test_flat = X_test_norm.reshape(X_test_norm.shape[0], -1)

        X_train_temporal = self.temporal_encoder.encode(X_train_flat)
        X_val_temporal = self.temporal_encoder.encode(X_val_flat)
        X_test_temporal = self.temporal_encoder.encode(X_test_flat)

        print(f"  Temporal shape: {X_train_temporal.shape}")

        # Step 7: Creazione TensorDatasets
        print("\n[6/6] Creating PyTorch datasets...")
        ds_train = TensorDataset(
            X_train_temporal, torch.tensor(y_train, dtype=torch.long)
        )
        ds_val = TensorDataset(X_val_temporal, torch.tensor(y_val, dtype=torch.long))
        ds_test = TensorDataset(X_test_temporal, torch.tensor(y_test, dtype=torch.long))

        # Metadata
        metadata = {
            "nb_inputs": self.feature_selector.nb_features,
            "nb_outputs": 3,
            "nb_steps": self.temporal_encoder.nb_steps,
            "dt": self.temporal_encoder.dt,
            "stress_type": self.stress_type,
            "normalization_params": norm_params,
            "selected_frequencies": self.feature_selector.selected_freq_indices,
            "split_strategy": "leave_one_plant_out",
            "leave_plant": leave_plant,
            "train_plants": np.unique(train_plants).tolist(),
            "val_test_split": {"val_size": val_size, "test_size": test_size},
        }

        print("\nâœ“ Leave-One-Plant-Out dataset preparation complete!")
        print(
            f"   Strategy: Training on {metadata['train_plants']}, Testing on {leave_plant}"
        )

        return ds_train, ds_val, ds_test, metadata
