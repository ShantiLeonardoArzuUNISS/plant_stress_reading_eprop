"""
plant_stress_detection_class.py

Complete Spiking Neural Network for Plant Stress Detection
Supports both BPTT and e-prop training methods
Configurable encoding strategies: Rate, Latency, Temporal Repetition

Author: Shanti Leonardo Arzu
Date: November 2025
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn

# Import dei moduli custom per gestione dati e encoding
from plant_data_management import PlantDataManager
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================================================
# SEZIONE 1: SURROGATE GRADIENT E FUNZIONI HELPER
# ============================================================================


class SurrGradSpike(torch.autograd.Function):
    """
    Funzione di attivazione spike con surrogate gradient.
    Implementa la normalized negative part of a fast sigmoid
    come in Zenke & Ganguli (2018).
    """

    scale = 10.0

    @staticmethod
    def forward(ctx, input, threshold=1.0):
        """Forward pass: step function"""
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        out = torch.zeros_like(input)
        out[input > threshold] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: surrogate gradient"""
        (input,) = ctx.saved_tensors
        threshold = ctx.threshold
        grad_input = grad_output.clone()
        grad = (
            grad_input / (SurrGradSpike.scale * torch.abs(input - threshold) + 1.0) ** 2
        )
        return grad, None


spike_fn = SurrGradSpike.apply


# ============================================================================
# SEZIONE 2: MODELLI NEURONI (CUBA LIF) - Riutilizzo dal progetto Braille
# ============================================================================


class CuBaLIF:
    """
    Current-Based Leaky Integrate-and-Fire neuron per layer feedforward.
    """

    def __init__(
        self,
        batch_size,
        nb_inputs,
        nb_neurons,
        alpha,
        beta,
        firing_threshold,
        fwd_scale=0.1,
        lower_bound=None,
        ref_per_timesteps=None,
        device="cuda",
        dtype=torch.float,
    ):
        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.alpha = alpha
        self.beta = beta
        self.firing_threshold = firing_threshold
        self.lower_bound = lower_bound
        self.ref_per_timesteps = ref_per_timesteps
        self.device = device
        self.dtype = dtype

        # Inizializza pesi feedforward
        self.ff_weights = torch.empty(
            (nb_inputs, nb_neurons), device=device, dtype=dtype, requires_grad=True
        )
        torch.nn.init.normal_(
            self.ff_weights, mean=0.0, std=fwd_scale / np.sqrt(nb_inputs)
        )

        # Inizializza stati
        self.reset_states(batch_size)

    def reset_states(self, batch_size):
        """Reset degli stati del layer"""
        self.syn = torch.zeros(
            (batch_size, self.nb_neurons), device=self.device, dtype=self.dtype
        )
        self.mem = torch.zeros(
            (batch_size, self.nb_neurons), device=self.device, dtype=self.dtype
        )
        self.rst = torch.zeros(
            (batch_size, self.nb_neurons), device=self.device, dtype=self.dtype
        )

        if self.ref_per_timesteps is not None:
            self.ref_per_counter = torch.zeros(
                (batch_size, self.nb_neurons), device=self.device, dtype=self.dtype
            )

    # ====================
    def reset(self):
        """Reset per compatibilità - chiama reset_states()"""
        batch_size = self.syn.shape[0]
        self.reset_states(batch_size)

    # =================================================

    def update_refractory_counter(self):
        """Aggiorna il contatore del periodo refrattario"""
        self.ref_per_counter = torch.clamp(self.ref_per_counter - 1, min=0)
        self.ref_per_counter = torch.where(
            self.rst > 0,
            torch.tensor(self.ref_per_timesteps, dtype=self.dtype, device=self.device),
            self.ref_per_counter,
        )

    def step(self, input_activity_t):
        """
        Computa l'attività del layer per un singolo timestep.
        """
        # Calcola input corrente
        h1 = torch.einsum("ab,bc->ac", input_activity_t, self.ff_weights)

        # Gestione periodo refrattario
        if self.ref_per_timesteps is not None:
            self.update_refractory_counter()
            mask_ready = (self.ref_per_counter == 0).float()
            self.syn = self.alpha * self.syn + h1 * mask_ready
        else:
            self.syn = self.alpha * self.syn + h1

        # Update membrana
        self.mem = (self.beta * self.mem + self.syn) * (1.0 - self.rst)

        # Clamping lower bound
        if self.lower_bound is not None:
            self.mem = torch.clamp(self.mem, min=self.lower_bound)

        # Spike generation
        mthr = self.mem - self.firing_threshold
        out = spike_fn(mthr, self.firing_threshold)
        self.rst = out.detach()

        return out, self.syn, self.mem


class CuBaRLIF:
    """
    Current-Based Recurrent Leaky Integrate-and-Fire neuron.
    """

    def __init__(
        self,
        batch_size,
        nb_inputs,
        nb_neurons,
        alpha,
        beta,
        firing_threshold,
        fwd_scale=0.1,
        rec_scale=0.1,
        lower_bound=None,
        ref_per_timesteps=None,
        device="cuda",
        dtype=torch.float,
    ):
        self.nb_inputs = nb_inputs
        self.nb_neurons = nb_neurons
        self.alpha = alpha
        self.beta = beta
        self.firing_threshold = firing_threshold
        self.lower_bound = lower_bound
        self.ref_per_timesteps = ref_per_timesteps
        self.device = device
        self.dtype = dtype

        # Inizializza pesi feedforward
        self.ff_weights = torch.empty(
            (nb_inputs, nb_neurons), device=device, dtype=dtype, requires_grad=True
        )
        torch.nn.init.normal_(
            self.ff_weights, mean=0.0, std=fwd_scale / np.sqrt(nb_inputs)
        )

        # Inizializza pesi ricorrenti
        self.rec_weights = torch.empty(
            (nb_neurons, nb_neurons), device=device, dtype=dtype, requires_grad=True
        )
        torch.nn.init.normal_(
            self.rec_weights, mean=0.0, std=rec_scale / np.sqrt(nb_neurons)
        )

        # Inizializza stati
        self.reset_states(batch_size)

    def reset_states(self, batch_size):
        """Reset degli stati del layer ricorrente"""
        self.syn = torch.zeros(
            (batch_size, self.nb_neurons), device=self.device, dtype=self.dtype
        )
        self.mem = torch.zeros(
            (batch_size, self.nb_neurons), device=self.device, dtype=self.dtype
        )
        self.rst = torch.zeros(
            (batch_size, self.nb_neurons), device=self.device, dtype=self.dtype
        )

        if self.ref_per_timesteps is not None:
            self.ref_per_counter = torch.zeros(
                (batch_size, self.nb_neurons), device=self.device, dtype=self.dtype
            )

    # ====================
    def reset(self):
        """Reset per compatibilità - chiama reset_states()"""
        batch_size = self.syn.shape[0]
        self.reset_states(batch_size)

    # =========================================================

    def update_refractory_counter(self):
        """Aggiorna il contatore del periodo refrattario"""
        self.ref_per_counter = torch.clamp(self.ref_per_counter - 1, min=0)
        self.ref_per_counter = torch.where(
            self.rst > 0,
            torch.tensor(self.ref_per_timesteps, dtype=self.dtype, device=self.device),
            self.ref_per_counter,
        )

    def step(self, input_activity_t):
        """
        Computa l'attività del layer ricorrente per un singolo timestep.
        """
        # Calcola contributo feedforward + ricorrente
        h1 = torch.einsum(
            "ab,bc->ac", input_activity_t, self.ff_weights
        ) + torch.einsum("ab,bc->ac", self.rst, self.rec_weights)

        # Gestione periodo refrattario
        if self.ref_per_timesteps is not None:
            self.update_refractory_counter()
            mask_ready = (self.ref_per_counter == 0).float()
            self.syn = self.alpha * self.syn + h1 * mask_ready
        else:
            self.syn = self.alpha * self.syn + h1

        # Update membrana
        self.mem = (self.beta * self.mem + self.syn) * (1.0 - self.rst)

        # Clamping lower bound
        if self.lower_bound is not None:
            self.mem = torch.clamp(self.mem, min=self.lower_bound)

        # Spike generation
        mthr = self.mem - self.firing_threshold
        out = spike_fn(mthr, self.firing_threshold)
        self.rst = out.detach()

        return out, self.syn, self.mem


# ============================================================================
# SEZIONE 3: CONFIGURAZIONE PARAMETRI
# ============================================================================


class ConfigParams:
    """
    Classe per gestire tutti i parametri configurabili del progetto.
    """

    def __init__(self):
        # ==================== PARAMETRI DATASET ====================
        self.stress_type = "water"  # 'water' o 'iron'
        self.data_file = "../data/Water_Stress.npz"

        # ==================== STRATEGIA DI SPLIT ====================
        self.split_strategy = "standard"  # 'standard' o 'leave_one_plant_out'
        self.train_size = 0.7
        self.val_size = 0.15
        self.test_size = 0.15
        self.leave_plant = "P3"  # Per Leave-One-Plant-Out
        self.lopo_val_size = 0.5
        self.lopo_test_size = 0.5

        # ==================== ENCODING PARAMETERS ====================
        self.encoding_type = "rate"  # 'rate', 'latency', 'temporal'
        self.nb_steps = 100
        self.dt = 1.0
        self.gain = 10.0

        # ==================== ARCHITETTURA RETE ====================
        self.nb_hidden = 100
        self.nb_outputs = 3

        # ==================== NEURONI PARAMETERS ====================
        self.tau_mem = 0.06
        self.tau_syn = 0.0
        self.tau_trace = 0.14
        self.tau_trace_out = 0.14
        self.firing_threshold = 1.0
        self.lower_bound = -1.0
        self.ref_per_timesteps = 3

        # ==================== TRAINING PARAMETERS ====================
        self.training_method = "bptt"  # 'bptt' o 'eprop'
        self.epochs = 40
        self.batch_size = 32
        self.batch_size_test = 32
        self.lr = 0.004
        self.gamma = 0.3

        # ==================== REGULARIZATION ====================
        self.reg_spikes = 0.001
        self.reg_neurons = 0.001

        # ==================== WEIGHT INITIALIZATION ====================
        self.fwd_weight_scale = 0.1
        self.weight_scale_factor = 1.0

        # ==================== DEVICE E SEED ====================
        self.use_seed = True
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ==================== PATHS ====================
        self.output_dir = "../results_plant_snn"
        self.figures_dir = os.path.join(self.output_dir, "figures")
        self.models_dir = os.path.join(self.output_dir, "models")

    def create_directories(self):
        """Crea le directory necessarie"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def save_config(self, filepath):
        """Salva configurazione in JSON"""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=4)

    def print_config(self):
        """Stampa configurazione"""
        print("\n" + "=" * 70)
        print("CONFIGURAZIONE PARAMETRI RETE")
        print("=" * 70)
        print(f"Dataset: {self.stress_type} stress - {self.data_file}")
        print(f"Split Strategy: {self.split_strategy}")
        if self.split_strategy == "standard":
            print(
                f"  Train/Val/Test: {self.train_size}/{self.val_size}/{self.test_size}"
            )
        else:
            print(f"  Leave-Out Plant: {self.leave_plant}")
        print(f"\nEncoding: {self.encoding_type}")
        print(f"  Steps: {self.nb_steps}, dt: {self.dt} ms, gain: {self.gain}")
        print("\nArchitettura:")
        print(f"  Hidden neurons: {self.nb_hidden}")
        print(f"  Output classes: {self.nb_outputs}")
        print("\nTraining:")
        print(f"  Method: {self.training_method.upper()}")
        print(f"  Epochs: {self.epochs}, Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.lr}")
        print(f"\nDevice: {self.device}")
        print("=" * 70 + "\n")


# ============================================================================
# SEZIONE 4: RETE NEURALE SPIKING (SRNN per Plant Stress)
# ============================================================================


class PlantStressSRNN:
    """
    Spiking Recurrent Neural Network per Plant Stress Detection.
    Supporta training con BPTT o e-prop.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.float

        # Calcola costanti temporali
        time_step = config.dt / 1000.0
        if config.tau_syn == 0.0:
            self.alpha = 0.0
        else:
            self.alpha = float(np.exp(-time_step / config.tau_syn))
        self.beta = float(np.exp(-time_step / config.tau_mem))
        self.beta_trace = float(np.exp(-time_step / config.tau_trace))
        self.beta_trace_out = float(np.exp(-time_step / config.tau_trace_out))

        print("\n[INFO] Costanti temporali calcolate:")
        print(f"  alpha (syn): {self.alpha:.4f}")
        print(f"  beta (mem): {self.beta:.4f}")
        print(f"  beta_trace: {self.beta_trace:.4f}")

        # Dimensioni (verranno settate dopo il caricamento dati)
        self.nb_inputs = None
        self.nb_hidden = config.nb_hidden
        self.nb_outputs = config.nb_outputs
        self.nb_steps = config.nb_steps

        # Layer
        self.rec_layer = None
        self.ff_layer = None

        # Storico training
        self.loss_hist = []
        self.train_acc_hist = []
        self.val_acc_hist = []

    def initialize_network(self, nb_inputs):
        """Inizializza i layer della rete"""
        self.nb_inputs = nb_inputs
        config = self.config

        print("\n[INFO] Inizializzazione rete:")
        print(f"  Input neurons: {self.nb_inputs}")
        print(f"  Hidden neurons: {self.nb_hidden}")
        print(f"  Output neurons: {self.nb_outputs}")

        # Layer ricorrente
        self.rec_layer = CuBaRLIF(
            batch_size=config.batch_size,
            nb_inputs=self.nb_inputs,
            nb_neurons=self.nb_hidden,
            alpha=self.alpha,
            beta=self.beta,
            firing_threshold=config.firing_threshold,
            fwd_scale=config.fwd_weight_scale,
            rec_scale=config.fwd_weight_scale * config.weight_scale_factor,
            lower_bound=config.lower_bound,
            ref_per_timesteps=config.ref_per_timesteps,
            device=self.device,
            dtype=self.dtype,
        )

        # Layer feedforward (readout)
        self.ff_layer = CuBaLIF(
            batch_size=config.batch_size,
            nb_inputs=self.nb_hidden,
            nb_neurons=self.nb_outputs,
            alpha=self.alpha,
            beta=self.beta,
            firing_threshold=config.firing_threshold,
            fwd_scale=config.fwd_weight_scale,
            lower_bound=config.lower_bound,
            ref_per_timesteps=config.ref_per_timesteps,
            device=self.device,
            dtype=self.dtype,
        )

        print("[INFO] ✓ Rete inizializzata con successo!")

    def forward(self, input_batch):
        """
        Forward pass attraverso la rete.

        Segue esattamente il pattern del codice Braille: pre-allocazione
        dei tensori e assegnazioni indicizzate invece di append + stack.
        """
        batch_size = input_batch.shape[0]
        nb_steps = input_batch.shape[1]

        # Reset stati con il batch size TOMATO
        self.rec_layer.reset_states(batch_size)  # ← USA batch_size ATTUALE!
        self.ff_layer.reset_states(batch_size)  # ← USA batch_size ATTUALE!

        # PRE-ALLOCAZIONE TENSORI (come in Braille)
        # Questo evita problemi con il computational graph
        rec_spk_rec = torch.zeros(
            (batch_size, nb_steps, self.nb_hidden), dtype=self.dtype, device=self.device
        )
        rec_syn_rec = torch.zeros(
            (batch_size, nb_steps, self.nb_hidden), dtype=self.dtype, device=self.device
        )
        rec_mem_rec = torch.zeros(
            (batch_size, nb_steps, self.nb_hidden), dtype=self.dtype, device=self.device
        )

        ff_spk_rec = torch.zeros(
            (batch_size, nb_steps, self.nb_outputs),
            dtype=self.dtype,
            device=self.device,
        )
        ff_syn_rec = torch.zeros(
            (batch_size, nb_steps, self.nb_outputs),
            dtype=self.dtype,
            device=self.device,
        )
        ff_mem_rec = torch.zeros(
            (batch_size, nb_steps, self.nb_outputs),
            dtype=self.dtype,
            device=self.device,
        )

        # Loop temporale (come in Braille)
        for t in range(nb_steps):
            # Hidden layer (ricorrente): input + ricorrenza
            rec_spk, rec_syn, rec_mem = self.rec_layer.step(input_batch[:, t, :])

            # ASSEGNAZIONE INDICIZZATA (NON APPEND!)
            rec_spk_rec[:, t, :] = rec_spk
            rec_syn_rec[:, t, :] = rec_syn
            rec_mem_rec[:, t, :] = rec_mem

            # Readout layer (feedforward): solo spikes da hidden
            ff_spk, ff_syn, ff_mem = self.ff_layer.step(rec_spk)

            # ASSEGNAZIONE INDICIZZATA (NON APPEND!)
            ff_spk_rec[:, t, :] = ff_spk
            ff_syn_rec[:, t, :] = ff_syn
            ff_mem_rec[:, t, :] = ff_mem

        # Return già nel formato corretto (no stack necessario)
        return [rec_spk_rec, rec_syn_rec, rec_mem_rec], [
            ff_spk_rec,
            ff_syn_rec,
            ff_mem_rec,
        ]

    def train_bptt(self, ds_train, ds_val, ds_test):
        """Training con Backpropagation Through Time"""
        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()

        generator = DataLoader(
            dataset=ds_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Pesi da ottimizzare
        layers = [
            self.rec_layer.ff_weights,
            self.rec_layer.rec_weights,
            self.ff_layer.ff_weights,
        ]

        optimizer = torch.optim.Adamax(layers, lr=self.config.lr, betas=(0.9, 0.995))

        best_val_acc = 0.0
        best_weights = None

        pbar_epochs = tqdm(range(self.config.epochs), desc="Training", position=0)
        for epoch in pbar_epochs:
            epoch_loss = []
            epoch_acc = []

            pbar_batches = tqdm(
                generator, desc=f"Epoch {epoch + 1}", leave=False, position=1
            )
            for x_batch, y_batch in pbar_batches:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                rec_outputs, ff_outputs = self.forward(x_batch)
                spk_rec_hidden = rec_outputs[0]
                spk_rec_readout = ff_outputs[0]

                # Loss calculation
                m = torch.sum(spk_rec_readout, 1)  # Sum spikes over time
                log_p_y = log_softmax_fn(m)

                # Regularization
                reg_loss = self.config.reg_spikes * torch.mean(
                    torch.sum(spk_rec_hidden, 1)
                )
                reg_loss += self.config.reg_neurons * torch.mean(
                    torch.sum(torch.sum(spk_rec_hidden, dim=0), dim=0) ** 2
                )

                # Total loss
                loss_val = loss_fn(log_p_y, y_batch) + reg_loss

                # Backward + optimization
                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                # Metriche
                epoch_loss.append(loss_val.item())
                _, predictions = torch.max(m, 1)
                acc = (y_batch == predictions).float().mean().item()
                epoch_acc.append(acc)

                pbar_batches.set_postfix(
                    {"loss": f"{loss_val.item():.4f}", "acc": f"{acc:.4f}"}
                )

            # Fine epoca: calcola metriche
            mean_loss = np.mean(epoch_loss)
            mean_train_acc = np.mean(epoch_acc)
            val_acc = self.compute_accuracy(ds_val)

            self.loss_hist.append(mean_loss)
            self.train_acc_hist.append(mean_train_acc)
            self.val_acc_hist.append(val_acc)

            # Salva best weights
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = {
                    "rec_ff": self.rec_layer.ff_weights.data.clone(),
                    "rec_rec": self.rec_layer.rec_weights.data.clone(),
                    "ff": self.ff_layer.ff_weights.data.clone(),
                }

            pbar_epochs.set_postfix(
                {
                    "train_acc": f"{mean_train_acc * 100:.2f}%",
                    "val_acc": f"{val_acc * 100:.2f}%",
                    "loss": f"{mean_loss:.4f}",
                }
            )

        return self.loss_hist, self.train_acc_hist, self.val_acc_hist, best_weights

    def compute_accuracy(self, dataset):
        """Calcola accuracy su un dataset"""
        generator = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size_test,
            shuffle=False,
            num_workers=0,
        )

        accs = []
        for x_batch, y_batch in generator:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                _, ff_outputs = self.forward(x_batch)
                spk_rec_readout = ff_outputs[0]
                m = torch.sum(spk_rec_readout, 1)
                _, predictions = torch.max(m, 1)
                acc = (y_batch == predictions).float().mean().item()
                accs.append(acc)

        return np.mean(accs)

    def compute_confusion_matrix(self, dataset, class_names):
        """Calcola confusion matrix"""
        generator = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size_test,
            shuffle=False,
            num_workers=0,
        )

        all_preds = []
        all_labels = []

        for x_batch, y_batch in generator:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                _, ff_outputs = self.forward(x_batch)
                spk_rec_readout = ff_outputs[0]
                m = torch.sum(spk_rec_readout, 1)
                _, predictions = torch.max(m, 1)

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        return cm, all_preds, all_labels


# ============================================================================
# SEZIONE 5: VISUALIZZAZIONE
# ============================================================================


def plot_training_curves(loss_hist, train_acc_hist, val_acc_hist, save_path):
    """Plotta le curve di training"""
    epochs = range(1, len(loss_hist) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, loss_hist, "b-", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Loss", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Train Accuracy
    axes[1].plot(
        epochs, np.array(train_acc_hist) * 100, "g-", linewidth=2, label="Train"
    )
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].set_title("Training Accuracy", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Validation Accuracy
    axes[2].plot(
        epochs, np.array(val_acc_hist) * 100, "r-", linewidth=2, label="Validation"
    )
    axes[2].set_xlabel("Epoch", fontsize=12)
    axes[2].set_ylabel("Accuracy (%)", fontsize=12)
    axes[2].set_title("Validation Accuracy", fontsize=14, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training curves salvate in: {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    """Plotta confusion matrix"""
    plt.figure(figsize=(8, 6))
    sn.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix salvata in: {save_path}")


# ============================================================================
# SEZIONE 6: MAIN
# ============================================================================


def main():
    """Funzione principale"""
    print("\n" + "=" * 70)
    print("PLANT STRESS DETECTION - SPIKING NEURAL NETWORK")
    print("=" * 70)

    # Configurazione
    config = ConfigParams()

    # Parsing argomenti opzionale
    parser = argparse.ArgumentParser(description="Plant Stress SNN Training")
    parser.add_argument(
        "--stress_type", type=str, default="water", choices=["water", "iron"]
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="../data/Water_Stress.npz",
        help="Path al file dataset .npz",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="standard",
        choices=["standard", "leave_one_plant_out"],
    )
    parser.add_argument(
        "--encoding", type=str, default="rate", choices=["rate", "latency", "temporal"]
    )
    parser.add_argument("--method", type=str, default="bptt", choices=["bptt", "eprop"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--nb_hidden", type=int, default=100)

    args = parser.parse_args()

    # Aggiorna config
    config.stress_type = args.stress_type
    config.data_file = args.data_file  # ← AGGIUNGI QUESTA RIGA
    config.split_strategy = args.split
    config.encoding_type = args.encoding
    config.training_method = args.method
    config.epochs = args.epochs
    config.lr = args.lr
    config.nb_hidden = args.nb_hidden

    config.create_directories()
    config.print_config()

    # Imposta seed
    if config.use_seed:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        print(f"[INFO] Seed impostato a: {config.seed}\n")

    # ========== CARICAMENTO DATASET ==========
    print("\n" + "=" * 70)
    print("CARICAMENTO E PREPROCESSING DATASET")
    print("=" * 70)

    encoding_params = {
        "encoding_type": config.encoding_type,
        "nb_steps": config.nb_steps,
        "dt": config.dt,
        "gain": config.gain,
    }

    data_manager = PlantDataManager(
        stress_type=config.stress_type, encoding_params=encoding_params
    )

    if config.split_strategy == "standard":
        ds_train, ds_val, ds_test, metadata = (
            data_manager.prepare_dataset_standard_split(
                file_path=config.data_file,
                train_size=config.train_size,
                val_size=config.val_size,
                test_size=config.test_size,
            )
        )
    else:
        ds_train, ds_val, ds_test, metadata = (
            data_manager.prepare_dataset_leave_one_plant_split(
                file_path=config.data_file,
                leave_plant=config.leave_plant,
                val_size=config.lopo_val_size,
                test_size=config.lopo_test_size,
            )
        )

    print("\n[INFO] Dataset preparato:")
    print(f"  Train samples: {len(ds_train)}")
    print(f"  Validation samples: {len(ds_val)}")
    print(f"  Test samples: {len(ds_test)}")
    print(f"  Input features: {metadata['nb_inputs']}")
    print(f"  Output classes: {metadata['nb_outputs']}")

    # ========== INIZIALIZZAZIONE RETE ==========
    print("\n" + "=" * 70)
    print("INIZIALIZZAZIONE RETE NEURALE")
    print("=" * 70)

    srnn = PlantStressSRNN(config)
    srnn.initialize_network(nb_inputs=metadata["nb_inputs"])

    # ========== TRAINING ==========
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    if config.training_method == "bptt":
        loss_hist, train_acc_hist, val_acc_hist, best_weights = srnn.train_bptt(
            ds_train, ds_val, ds_test
        )
    else:
        print("[WARNING] e-prop non ancora implementato. Uso BPTT.")
        loss_hist, train_acc_hist, val_acc_hist, best_weights = srnn.train_bptt(
            ds_train, ds_val, ds_test
        )

    # ========== VALUTAZIONE FINALE ==========
    print("\n" + "=" * 70)
    print("VALUTAZIONE FINALE")
    print("=" * 70)

    # Carica best weights
    srnn.rec_layer.ff_weights.data = best_weights["rec_ff"]
    srnn.rec_layer.rec_weights.data = best_weights["rec_rec"]
    srnn.ff_layer.ff_weights.data = best_weights["ff"]

    final_train_acc = srnn.compute_accuracy(ds_train)
    final_val_acc = srnn.compute_accuracy(ds_val)
    final_test_acc = srnn.compute_accuracy(ds_test)

    print("\n[RISULTATI FINALI]")
    print(f"  Train Accuracy: {final_train_acc * 100:.2f}%")
    print(f"  Validation Accuracy: {final_val_acc * 100:.2f}%")
    print(f"  Test Accuracy: {final_test_acc * 100:.2f}%")

    # Confusion matrix
    cm_test, preds, labels = srnn.compute_confusion_matrix(
        ds_test, class_names=["Healthy", "Mild", "Severe"]
    )

    print("\n[CONFUSION MATRIX - TEST SET]")
    print(cm_test)

    # Classification report
    report = classification_report(
        labels, preds, target_names=["Healthy", "Mild", "Severe"], digits=4
    )
    print("\n[CLASSIFICATION REPORT]")
    print(report)

    # ========== SALVATAGGIO ==========
    print("\n" + "=" * 70)
    print("SALVATAGGIO RISULTATI")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Config
    config_path = os.path.join(config.output_dir, f"config_{timestamp}.json")
    config.save_config(config_path)

    # Training curves
    curves_path = os.path.join(config.figures_dir, f"training_curves_{timestamp}.png")
    plot_training_curves(loss_hist, train_acc_hist, val_acc_hist, curves_path)

    # Confusion matrix
    cm_path = os.path.join(config.figures_dir, f"confusion_matrix_{timestamp}.png")
    plot_confusion_matrix(cm_test, ["Healthy", "Mild", "Severe"], cm_path)

    # Modello
    model_path = os.path.join(config.models_dir, f"best_model_{timestamp}.pt")
    torch.save(
        {
            "config": config.__dict__,
            "weights": best_weights,
            "metadata": metadata,
            "train_acc": final_train_acc,
            "val_acc": final_val_acc,
            "test_acc": final_test_acc,
            "loss_hist": loss_hist,
            "train_acc_hist": train_acc_hist,
            "val_acc_hist": val_acc_hist,
        },
        model_path,
    )
    print(f"[INFO] Modello salvato in: {model_path}")

    # Risultati CSV
    results_csv = os.path.join(config.output_dir, f"results_{timestamp}.csv")
    results_df = pd.DataFrame(
        {
            "Epoch": range(1, len(loss_hist) + 1),
            "Loss": loss_hist,
            "Train_Accuracy": train_acc_hist,
            "Val_Accuracy": val_acc_hist,
        }
    )
    results_df.to_csv(results_csv, index=False)
    print(f"[INFO] Risultati salvati in: {results_csv}")

    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETATA CON SUCCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    main()
