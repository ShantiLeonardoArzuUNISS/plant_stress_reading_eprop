"""
plant_data_preprocessing.py

Feature selection, normalization.

Author: Shanti Leonardo Arzu
Date: November 2025
"""

import numpy as np


class PlantFeatureSelector:
    """
    Gestisce la selezione delle frequenze e normalizzazione dei dati di bioimpedenza
    """

    def __init__(self, stress_type="water", custom_freq_indices=None):
        """
        Args:
            stress_type (str): 'water', 'iron', o 'custom'
            custom_freq_indices (list): Indici custom se stress_type ='custom'
        """
        self.stress_type = stress_type

        # Definisci indici frequenze ottimali (da Elastic Net)
        if stress_type == "water":
            self.selected_freq_indices = [0, 1, 2]  # 100-120 Hz
            print("[INFO] Water stress: usando frequenze 100-120 Hz (indici [0, 1, 2])")
        elif stress_type == "iron":
            self.selected_freq_indices = [197, 198, 199]  # 4.7-10 MHz
            print(
                "[INFO] Iron stress: usando frequenze 4.7-10 MHz (indici [197, 198, 199])"
            )
        elif stress_type == "custom" and custom_freq_indices is not None:
            self.selected_freq_indices = custom_freq_indices
            print(
                f"[INFO] Custom stress: usando {len(custom_freq_indices)} frequenze custom"
            )
        else:
            raise ValueError(
                "stress_type deve essere 'water', 'iron' o 'custom' (con custom_freq_indices)"
            )

        self.nb_selected_freqs = len(self.selected_freq_indices)
        self.nb_features = self.nb_selected_freqs * 2  # real + imag

        print(
            f"[INFO] Configurazione: {self.nb_selected_freqs} frequenze → {self.nb_features} features"
        )

    def select_features(self, X):
        """
        Seleziona le frequenze rilevanti dal dataset completo

        Args:
            X (np.ndarray): Dati in formato flat (n_samples, 400)
                            o shaped (n_samples, 200, 2)

        Returns:
            X_selected (np.ndarray): (n_samples, nb_selected_freqs, 2)
        """
        # Converti in formato shaped se necessario
        if X.ndim == 2 and X.shape[1] == 400:
            print(f"[INFO] Conversione Flattened → Shaped: {X.shape} → ", end="")
            X = X.reshape(-1, 200, 2)
            print(f"{X.shape}")

        # Verifica formato corretto
        if X.ndim != 3 or X.shape[1] != 200 or X.shape[2] != 2:
            raise ValueError(
                f"Input deve essere (n_samples, 200, 2), ricevuto: {X.shape}"
            )

        # Seleziona frequenze
        X_selected = X[:, self.selected_freq_indices, :]

        print(f"[INFO] Feature reduction: {X.shape} → {X_selected.shape}")
        print(f"      Ridotte da 200 a {self.nb_selected_freqs} frequenze")
        print(f"      Features totali: 400 → {self.nb_features}")

        return X_selected

    def normalize_features(self, X_train, X_val, X_test):
        """
        Normalizza usando Z-score (media e std calcolati SOLO su training)

        Args:
            X_train (np.ndarray): Training set (n_train, nb_freqs, 2)
            X_val (np.ndarray): Validation set (n_val, nb_freqs, 2)
            X_test (np.ndarray): Test set (n_test, nb_freqs, 2)

        Returns:
            X_train_norm, X_val_norm, X_test_norm: Array normalizzati
            norm_params (dict): {'mean': ..., 'std': ...}
        """
        # Calcola mean e std SOLO su training (evita data leakage)
        mean = np.mean(X_train, axis=0, keepdims=True)  # (1, nb_freqs, 2)
        std = np.std(X_train, axis=0, keepdims=True)  # (1, nb_freqs, 2)

        # Evita divisione per zero
        std[std == 0] = 1.0

        # Normalizza tutti i set usando i parametri del training
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        X_test_norm = (X_test - mean) / std

        norm_params = {"mean": mean, "std": std}

        print("[INFO] Normalizzazione completata:")
        print(f"      Mean shape: {mean.shape}")
        print(f"      Std shape: {std.shape}")
        print(
            f"      Range normalizzato: [{X_train_norm.min():.2f}, {X_train_norm.max():.2f}]"
        )

        return X_train_norm, X_val_norm, X_test_norm, norm_params
