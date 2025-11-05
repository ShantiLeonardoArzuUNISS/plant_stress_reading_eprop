"""
plant_data_preprocessing.py - Versione Migliorata

Feature selection, normalization, and dataset splitting for plant bioimpedance data.

Author: Shanti Leonardo Arzu
Date: November 2025
"""

import numpy as np
from sklearn.model_selection import train_test_split


class PlantFeatureSelector:
    """
    Gestisce la selezione delle frequenze, normalizzazione e splitting del dataset
    per i dati di bioimpedenza delle piante (water stress o iron stress).

    Questa classe offre massima flessibilità nella selezione delle frequenze:
    - Usa indici predefiniti da Elastic Net per "water" o "iron"
    - Permette selezione custom tramite array di indici
    """

    def __init__(self, stress_type="water", custom_freq_indices=None):
        """
        Inizializza il selettore di features.
        Args:
            stress_type (str): Tipo di stress da analizzare
                - "water": Usa frequenze 100-120 Hz (indici [0, 1, 2])
                - "iron": Usa frequenze 4.7-10 MHz (indici [197, 198, 199])
                - "custom": Usa indici personalizzati forniti in custom_freq_indices

            custom_freq_indices (list, optional): Lista di indici custom delle frequenze.
                Esempio: [5, 10, 15, 20] seleziona 4 frequenze → 8 features totali
                Se None, usa i default basati su stress_type.

        Raises:
            ValueError: Se stress_type non è valido o se "custom" senza custom_freq_indices
        """
        self.stress_type = stress_type

        # Selezione delle frequenze ottimali
        if stress_type == "water":
            # Water Stress: 100-120 Hz (feature indices 0-2)
            # Identificato da Elastic Net come ottimale per stress idrico
            self.selected_freq_indices = [0, 1, 2]
            print(
                f"[INFO] Water stress: usando frequenze 100-120 Hz (indici {self.selected_freq_indices})"
            )

        elif stress_type == "iron":
            # Iron Stress: 4.7-10 MHz (feature indices 197-199)
            # Identificato da Elastic Net come ottimale per carenza di ferro
            self.selected_freq_indices = [197, 198, 199]
            print(
                f"[INFO] Iron stress: usando frequenze 4.7-10 MHz (indici {self.selected_freq_indices})"
            )

        elif stress_type == "custom":
            # Custom frequencies: usa indici forniti dall'utente
            if custom_freq_indices is None:
                raise ValueError(
                    "stress_type='custom' richiede custom_freq_indices. "
                    "Fornisci una lista di indici, es: [5, 10, 15]"
                )
            if not isinstance(
                custom_freq_indices, (list, np.ndarray)
            ):  # se non è una lista o un array numpy errore
                raise ValueError(
                    f"custom_freq_indices deve essere list o np.ndarray, "
                    f"ricevuto {type(custom_freq_indices)}"
                )
            self.selected_freq_indices = list(custom_freq_indices)
            print(
                f"[INFO] Custom frequencies: usando indici {self.selected_freq_indices}"
            )

        else:
            raise ValueError(
                f"stress_type non valido: '{stress_type}'. "
                f"Usa 'water', 'iron', o 'custom'."
            )

        # Calcola numero di features risultanti
        self.num_frequencies = len(self.selected_freq_indices)
        self.num_features = self.num_frequencies * 2  # Real + Imaginary per ogni freq

        print(
            f"[INFO] Configurazione: {self.num_frequencies} frequenze → {self.num_features} features"
        )

    def flatten_to_shaped(self, X):
        """
        Converte dataset da formato Flattened a Shaped.

        Args:
            X (np.ndarray): Dataset in formato:
                - Flattened: (n_samples, 400) - [R0, I0, R1, I1, ..., R199, I199]
                - Shaped: (n_samples, 200, 2) - [[R0, I0], [R1, I1], ...]
        Returns:
            np.ndarray: Dataset in formato Shaped (n_samples, 200, 2)
        Note:
            Se l'input è già in formato Shaped, lo restituisce invariato.
        """
        # Verifica se è già in formato Shaped
        if len(X.shape) == 3 and X.shape[2] == 2:
            print(f"[INFO] Dataset già in formato Shaped: {X.shape}")
            return X

        # Conversione da Flattened (n_samples, 400) a Shaped (n_samples, 200, 2)
        if len(X.shape) == 2 and X.shape[1] == 400:
            X_shaped = X.reshape(-1, 200, 2)
            print(
                f"[INFO] Conversione Flattened → Shaped: {X.shape} → {X_shaped.shape}"
            )
            return X_shaped

        # Formato non riconosciuto
        raise ValueError(
            f"Formato non supportato: {X.shape}. "
            f"Atteso (n_samples, 400) o (n_samples, 200, 2)"
        )

    def reduce_dataset(self, X):
        """
        Riduce il dataset selezionando solo le frequenze specificate in self.selected_freq_indices.

        Args:
            X (np.ndarray): Dataset completo in formato:
                - Flattened: (n_samples, 400)
                - Shaped: (n_samples, 200, 2)
        Returns:
            np.ndarray: Dataset ridotto (n_samples, num_frequencies, 2)
                Esempio: con 3 frequenze → (n_samples, 3, 2)
        Esempio:
            >>> selector = PlantFeatureSelector(stress_type="water") # creo la classe
            >>> X_full = np.random.randn(2016, 400) # creo un dataset random per testare la funzione
            >>> X_reduced = selector.reduce_dataset(X_full) # riduco il dataset
            >>> print(X_reduced.shape)  # (2016, 3, 2)
        """
        # Step 1: Converti a formato Shaped se necessario
        X_shaped = self.flatten_to_shaped(X)

        # Step 2: Selezione delle frequenze tramite indicizzazione avanzata
        X_reduced = X_shaped[:, self.selected_freq_indices, :]

        print(f"[INFO] Feature reduction: {X_shaped.shape} → {X_reduced.shape}")
        print(f"      Ridotte da {X_shaped.shape[1]} a {X_reduced.shape[1]} frequenze")
        print(
            f"      Features totali: {X_shaped.shape[1] * 2} → {X_reduced.shape[1] * 2}"
        )

        return X_reduced

    def compute_statistics(self, X):
        """
        Calcola direttamente media e deviazione standard di un dataset.

        Args:
            X (np.ndarray): Dataset (n_samples, n_frequencies, 2)

        Returns:
            mean (np.ndarray): Media per ogni feature (1, n_frequencies, 2)
            std (np.ndarray): Deviazione standard per ogni feature (1, n_frequencies, 2)

        Note:
            Le statistiche sono calcolate lungo axis=0 (su tutti i campioni).
            keepdims=True mantiene le dimensioni per broadcasting durante normalizzazione.

        Esempio:
            >>> selector = PlantFeatureSelector(stress_type="water")
            >>> X_train = np.random.randn(1411, 3, 2)
            >>> mean, std = selector.compute_statistics(X_train)
            >>> print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
            Mean shape: (1, 3, 2), Std shape: (1, 3, 2)
        """
        # Calcola media lungo axis=0 (su tutti i campioni)
        mean = np.mean(X, axis=0, keepdims=True)  # (1, n_frequencies, 2)

        # Calcola deviazione standard lungo axis=0
        std = np.std(X, axis=0, keepdims=True)  # (1, n_frequencies, 2)

        # Protezione divisione per zero
        # Se una feature ha std=0 (tutti i valori identici), sostituisci con 1.0
        std[std == 0] = 1.0

        return mean, std

    def normalize_dataset(self, X):
        """
        Normalizza un singolo dataset usando Z-score normalization.
        Args:
            X (np.ndarray): Dataset da normalizzare (n_samples, n_frequencies, 2)
        Returns:
            X_norm (np.ndarray): Dataset normalizzato (n_samples, n_frequencies, 2)
            norm_params (dict): Parametri di normalizzazione {'mean': mean, 'std': std}
        Note:
            Mean e std sono calcolati lungo axis=0 (su tutti i campioni).
            Per normalizzare train/val/test coerentemente, usa normalize_features().
        """
        mean, std = self.compute_statistics(
            X
        )  # Metodo compute_statistics generato da me

        # Normalizza
        X_norm = (X - mean) / std

        # Parametri di normalizzazione
        norm_params = {"mean": mean, "std": std}

        print("[INFO] Normalizzazione Z-score completata")
        print(f"      Mean shape: {mean.shape}, Std shape: {std.shape}")

        return X_norm, norm_params

    def normalize_features(self, X_train, X_test, X_val=None):
        """
        Normalizza train/test/val usando SOLO le statistiche del training set.

        Args:
            X_train (np.ndarray): Training data (n_train, n_frequencies, 2)
            X_test (np.ndarray): Test data (n_test, n_frequencies, 2)
            X_val (np.ndarray, optional): Validation data (n_val, n_frequencies, 2)

        Returns:
            X_train_norm (np.ndarray): Training normalizzato
            X_test_norm (np.ndarray): Test normalizzato
            X_val_norm (np.ndarray or None): Validation normalizzato (se fornito)
            normalization_params (dict): {'mean': mean, 'std': std}

        Note:
            È FONDAMENTALE calcolare mean/std SOLO sul training set per evitare data leakage.
            Gli stessi parametri vengono applicati a validation e test set.
        """
        print("\n[INFO] === Normalizzazione Features ===")

        # Step 1: Calcola statistiche SOLO su training set usando il nuovo metodo
        mean, std = self.compute_statistics(X_train)

        print("[INFO] Statistiche calcolate su training set:")
        print(f"      Train samples: {X_train.shape[0]}")
        print(f"      Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
        print(f"      Std range: [{std.min():.4f}, {std.max():.4f}]")

        # Step 2: Applica normalizzazione a tutti i set
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

        print("[INFO] Normalizzazione applicata:")
        print(
            f"      Train: {X_train.shape} → mean={X_train_norm.mean():.4f}, std={X_train_norm.std():.4f}"
        )
        print(
            f"      Test: {X_test.shape} → mean={X_test_norm.mean():.4f}, std={X_test_norm.std():.4f}"
        )

        # Step 3: Normalizza validation se presente
        X_val_norm = None
        if X_val is not None:
            X_val_norm = (X_val - mean) / std
            print(
                f"      Val: {X_val.shape} → mean={X_val_norm.mean():.4f}, std={X_val_norm.std():.4f}"
            )

        # Step 4: Salva parametri
        normalization_params = {"mean": mean, "std": std}

        return X_train_norm, X_test_norm, X_val_norm, normalization_params

    def split_dataset(
        self, X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
    ):
        """
        Suddivide il dataset in train/validation/test con stratificazione.

        Args:
            X (np.ndarray): Features complete (n_samples, n_frequencies, 2)
            y (np.ndarray): Labels (n_samples,)
            train_size (float): Proporzione training set (default 0.7 = 70%)
            val_size (float): Proporzione validation set (default 0.15 = 15%)
            test_size (float): Proporzione test set (default 0.15 = 15%)
            random_state (int): Seed per riproducibilità (default 42)

        Returns:
            X_train, X_val, X_test (np.ndarray): Features splittate
            y_train, y_val, y_test (np.ndarray): Labels splittate

        Note:
            - La somma di train_size + val_size + test_size deve essere 1.0
            - Usa stratify per mantenere proporzioni di classe in ogni split

        Esempio:
            >>> X = np.random.randn(2016, 3, 2)
            >>> y = np.random.randint(0, 3, 2016)
            >>> selector = PlantFeatureSelector(stress_type="water")
            >>> X_train, X_val, X_test, y_train, y_val, y_test = selector.split_dataset(X, y)
            >>> print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            Train: 1411, Val: 302, Test: 303
        """
        print("\n[INFO] === Dataset Splitting ===")

        # Validazione proporzioni
        total_size = train_size + val_size + test_size
        if not np.isclose(total_size, 1.0):
            raise ValueError(
                f"train_size + val_size + test_size deve essere 1.0, "
                f"ricevuto {total_size:.3f}"
            )

        # Step 1: Split train vs temp (val + test)
        temp_size = val_size + test_size
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=temp_size,
            shuffle=True,
            stratify=y,
            random_state=random_state,
        )

        print("[INFO] Step 1 - Train vs Temp split:")
        print(f"      Train: {X_train.shape[0]} samples ({train_size * 100:.1f}%)")
        print(f"      Temp: {X_temp.shape[0]} samples ({temp_size * 100:.1f}%)")

        # Step 2: Split temp → val/test (50/50 del temp)
        val_fraction = val_size / temp_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1 - val_fraction),
            shuffle=True,
            stratify=y_temp,
            random_state=random_state,
        )

        print("[INFO] Step 2 - Val vs Test split:")
        print(f"      Val: {X_val.shape[0]} samples ({val_size * 100:.1f}%)")
        print(f"      Test: {X_test.shape[0]} samples ({test_size * 100:.1f}%)")

        # Verifica distribuzione classi
        print("\n[INFO] Distribuzione classi:")
        print(f"      Train: {np.bincount(y_train)}")
        print(f"      Val: {np.bincount(y_val)}")
        print(f"      Test: {np.bincount(y_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test
