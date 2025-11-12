"""
temporal_encoding.py
Temporal encoding strategies for static plant bioimpedance data

Author: Shanti Leonardo Arzu
Date: November 2025
"""

import numpy as np
import torch


class TemporalEncoder:
    """
    Trasforma dati statici in sequenze temporali per SNN.
    """

    def __init__(self, encoding_type="rate", nb_steps=100, dt=1.0, gain=10.0):
        """
        Args:
            encoding_type (str): 'rate', 'latency', 'temporal'
            nb_steps (int): Numero di timesteps nella sequenza
            dt (float): Durata del timestep in ms
            gain (float): Moltiplicatore per rate encoding (default: 10.0)
        """
        self.encoding_type = encoding_type
        self.nb_steps = nb_steps
        self.dt = dt
        self.gain = gain  # salvo gain come attributo

    def rate_encoding(self, X):
        """
        Rate Encoding: converte valori continui in rate di spike.

        Args:
            X (np.ndarray): Features normalizzate (n_samples, n_features)

        Returns:
            spikes (torch.Tensor): (n_samples, nb_steps, n_features)
        """
        n_samples, n_features = X.shape

        # Shift per rendere valori positivi (normalizzati sono ~[-3, +3])
        X_positive = X + 3.0  # Ora ~[0, 6]

        # Calcola probabilità di spike usando self.gain
        spike_prob = np.clip((X_positive * self.gain) / 100.0, 0, 1)

        # Genera spike trains stocastici (Poisson-like)
        spikes = np.zeros((n_samples, self.nb_steps, n_features))

        for t in range(self.nb_steps):
            random_values = np.random.rand(n_samples, n_features)
            spikes[:, t, :] = (random_values < spike_prob).astype(float)

        return torch.tensor(spikes, dtype=torch.float32)

    def latency_encoding(self, X):
        """
        Latency Encoding: valori alti → spike precoce, valori bassi → spike tardivo

        Args:
            X (np.ndarray): Features normalizzate (n_samples, n_features)

        Returns:
            spikes (torch.Tensor): (n_samples, nb_steps, n_features)
        """
        n_samples, n_features = X.shape

        # Shift e normalizza in [0, 1]
        X_shifted = X + 3.0  # ~[0, 6]
        X_norm = np.clip(X_shifted / 6.0, 0, 1)

        # Converti in latenza: valore alto → timestep basso
        latencies = ((1.0 - X_norm) * (self.nb_steps - 1)).astype(int)

        # Genera spikes
        spikes = np.zeros((n_samples, self.nb_steps, n_features))

        for i in range(n_samples):
            for j in range(n_features):
                t_spike = latencies[i, j]
                if 0 <= t_spike < self.nb_steps:
                    spikes[i, t_spike, j] = 1.0

        return torch.tensor(spikes, dtype=torch.float32)

    def temporal_repetition_encoding(self, X):
        """
        Temporal Repetition: ripeti il valore normalizzato per nb_steps timesteps

        Args:
            X (np.ndarray): Features normalizzate (n_samples, n_features)

        Returns:
            repeated (torch.Tensor): (n_samples, nb_steps, n_features)
        """
        n_samples, n_features = X.shape

        # Shift per rendere positivi
        X_positive = X + 3.0

        # Normalizza in [0, 1]
        X_norm = np.clip(X_positive / 6.0, 0, 1)

        # Ripeti il valore su tutti i timesteps
        repeated = np.tile(X_norm[:, np.newaxis, :], (1, self.nb_steps, 1))

        return torch.tensor(repeated, dtype=torch.float32)

    def encode(self, X):
        """
        Metodo principale: applica l'encoding scelto

        Args:
            X (np.ndarray): Features normalizzate (n_samples, n_features)

        Returns:
            encoded (torch.Tensor): (n_samples, nb_steps, n_features)
        """
        if self.encoding_type == "rate":
            return self.rate_encoding(X)
        elif self.encoding_type == "latency":
            return self.latency_encoding(X)
        elif self.encoding_type == "temporal":
            return self.temporal_repetition_encoding(X)
        else:
            raise ValueError(
                f"Encoding type '{self.encoding_type}' non supportato. "
                f"Scegli tra: 'rate', 'latency', 'temporal'"
            )
