"""
plot_normalized_standard.py

Script per visualizzare i dati di bioimpedenza delle piante NORMALIZZATI
con split standard 70/15/15 (train/val/test).
La normalizzazione usa media e std del 70% dei dati di training.

Author: Shanti Leonardo Arzu
Date: November 2025
"""

import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split

# Aggiungi il path parent per importare i moduli
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data(stress_type):
    """
    Carica i dati dal file .npz
    """
    file_path = f"../data/{stress_type.capitalize()}_Stress.npz"

    if not os.path.exists(file_path):
        print(f"❌ File non trovato: {file_path}")
        print("   Assicurati che il file sia nella cartella 'data/'")
        return None, None, None

    data = np.load(file_path, allow_pickle=True)
    X = data["X"]  # (2016, 400)
    y = data["y"]  # (2016,)
    plant_ids = data["plant_ids"]  # (2016,)

    print(f"\n✓ Dataset caricato: {stress_type.upper()}_Stress.npz")
    print(f"  Shape: {X.shape}")
    print(f"  Classi: {np.unique(y)} (0=Control, 1=Early, 2=Late)")
    print(f"  Piante: {np.unique(plant_ids)}")

    return X, y, plant_ids


def normalize_data_standard_split(X, y, train_size=0.7, val_size=0.15, test_size=0.15):
    """
    Normalizza i dati usando split standard 70/15/15
    Calcola media e std SOLO sul 70% dei dati di training
    """
    # Reshape da flattened a structured
    X_shaped = X.reshape(-1, 200, 2)  # (n_samples, n_frequencies, 2)

    # Split train/temp (70/30)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_shaped,
        y,
        test_size=(val_size + test_size),
        shuffle=True,
        stratify=y,
        random_state=42,
    )

    # Split temp → val/test (50/50 del 30%)
    val_fraction = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - val_fraction),
        shuffle=True,
        stratify=y_temp,
        random_state=42,
    )

    print("\n📊 Split dataset:")
    print(f"   Train: {len(X_train)} samples (70%)")
    print(f"   Val:   {len(X_val)} samples (15%)")
    print(f"   Test:  {len(X_test)} samples (15%)")

    # CALCOLA STATISTICHE SOLO SUL TRAINING SET (70%)
    mean_train = np.mean(X_train, axis=0, keepdims=True)  # (1, 200, 2)
    std_train = np.std(X_train, axis=0, keepdims=True)  # (1, 200, 2)

    # Evita divisione per zero
    std_train[std_train == 0] = 1.0

    print("\n📊 Statistiche calcolate sul 70% di training:")
    print(f"   Mean range: [{mean_train.min():.4f}, {mean_train.max():.4f}]")
    print(f"   Std range: [{std_train.min():.4f}, {std_train.max():.4f}]")

    # NORMALIZZA TUTTO IL DATASET CON LE STATISTICHE DEL TRAINING
    X_normalized = (X_shaped - mean_train) / std_train

    return X_normalized, y, mean_train, std_train


def prepare_data(X_normalized):
    """
    Prepara i dati normalizzati per la visualizzazione
    """
    # Frequenze logaritmiche da 100 Hz a 10 MHz
    frequencies = np.logspace(np.log10(100), np.log10(1e7), 200)
    return frequencies


def plot_2d_normalized(X_normalized, y, frequencies, stress_type):
    """
    Crea plot 2D dei dati NORMALIZZATI
    """
    # Colori per le classi
    colors = {
        0: "#2ecc71",  # Verde per Control
        1: "#f39c12",  # Arancione per Early Stress
        2: "#e74c3c",  # Rosso per Late Stress
    }

    class_names = {0: "Control", 1: "Early_Stress", 2: "Late_Stress"}

    # Crea subplot con 2 righe (Real e Imaginary)
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Full Spectrum - Real Part per Classe",
            "Full Spectrum - Imaginary Part per Classe",
        ),
        vertical_spacing=0.15,
    )

    # Per ogni classe
    for cls in sorted(np.unique(y)):
        mask = y == cls
        class_data = X_normalized[mask]

        # Calcola media e deviazione standard
        real_mean = np.mean(class_data[:, :, 0], axis=0)
        real_std = np.std(class_data[:, :, 0], axis=0)

        imag_mean = np.mean(class_data[:, :, 1], axis=0)
        imag_std = np.std(class_data[:, :, 1], axis=0)

        # Plot Real Part
        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=real_mean,
                mode="lines",
                name=class_names[cls],
                line=dict(color=colors[cls], width=3),
                legendgroup=class_names[cls],
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Aggiungi banda di confidenza (std) per Real
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([frequencies, frequencies[::-1]]),
                y=np.concatenate([real_mean + real_std, (real_mean - real_std)[::-1]]),
                fill="toself",
                fillcolor=colors[cls],
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                legendgroup=class_names[cls],
                opacity=0.2,
            ),
            row=1,
            col=1,
        )

        # Plot Imaginary Part
        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=imag_mean,
                mode="lines",
                name=class_names[cls],
                line=dict(color=colors[cls], width=3),
                legendgroup=class_names[cls],
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Aggiungi banda di confidenza (std) per Imaginary
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([frequencies, frequencies[::-1]]),
                y=np.concatenate([imag_mean + imag_std, (imag_mean - imag_std)[::-1]]),
                fill="toself",
                fillcolor=colors[cls],
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                legendgroup=class_names[cls],
                opacity=0.2,
            ),
            row=2,
            col=1,
        )

    # Update axes
    fig.update_xaxes(
        title_text="Frequenza (Hz)", type="log", row=1, col=1, gridcolor="lightgray"
    )
    fig.update_xaxes(
        title_text="Frequenza (Hz)", type="log", row=2, col=1, gridcolor="lightgray"
    )
    fig.update_yaxes(
        title_text="Real Part (Media ± Std)", row=1, col=1, gridcolor="lightgray"
    )
    fig.update_yaxes(
        title_text="Imaginary Part (Media ± Std)", row=2, col=1, gridcolor="lightgray"
    )

    # Update layout
    fig.update_layout(
        title=f"Trend di Frequenza (Full Spectrum) [Z-SCORE NORMALIZED]<br><sub>Stress Type: {stress_type.upper()} | Standard Split 70/15/15</sub>",
        height=800,
        showlegend=True,
        legend=dict(x=1.02, y=1),
        hovermode="x",
        template="plotly_white",
    )

    return fig


def plot_3d_normalized(X_normalized, y, frequencies, stress_type):
    """
    Crea plot 3D dei dati NORMALIZZATI
    """
    # Colori per le classi
    colors = {
        0: "#2ecc71",  # Verde per Control
        1: "#f39c12",  # Arancione per Early Stress
        2: "#e74c3c",  # Rosso per Late Stress
    }

    class_names = {0: "Control", 1: "Early_Stress", 2: "Late_Stress"}

    fig = go.Figure()

    # Per ogni classe
    for cls in sorted(np.unique(y)):
        mask = y == cls
        class_data = X_normalized[mask]

        # Calcola media per ogni frequenza
        real_mean = np.mean(class_data[:, :, 0], axis=0)
        imag_mean = np.mean(class_data[:, :, 1], axis=0)

        # Frequenze in scala logaritmica per visualizzazione
        freq_log = np.log10(frequencies)

        fig.add_trace(
            go.Scatter3d(
                x=real_mean,
                y=imag_mean,
                z=freq_log,
                mode="lines+markers",
                name=class_names[cls],
                line=dict(color=colors[cls], width=4),
                marker=dict(size=2, color=colors[cls]),
            )
        )

    # Update layout
    fig.update_layout(
        title=f"3D Trend di Frequenza: Real vs Imaginary vs Frequency [Z-SCORE NORMALIZED]<br><sub>Stress Type: {stress_type.upper()} | Standard Split 70/15/15</sub>",
        scene=dict(
            xaxis_title="Real Part (Media)",
            yaxis_title="Imaginary Part (Media)",
            zaxis_title="Frequency (log10 scale)",
            xaxis=dict(gridcolor="lightgray"),
            yaxis=dict(gridcolor="lightgray"),
            zaxis=dict(gridcolor="lightgray"),
        ),
        height=700,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )

    return fig


def main():
    """
    Funzione principale
    """
    print("\n" + "=" * 70)
    print("  VISUALIZZAZIONE DATI NORMALIZZATI (SPLIT STANDARD 70/15/15)")
    print("=" * 70)

    # Scelta del dataset
    while True:
        stress_type = input(
            "\n📊 Quale dataset vuoi visualizzare?\n   1) Water Stress\n   2) Iron Stress\n   Scelta (1/2): "
        ).strip()

        if stress_type == "1":
            stress_type = "water"
            break
        elif stress_type == "2":
            stress_type = "iron"
            break
        else:
            print("❌ Scelta non valida. Inserisci 1 o 2.")

    # Scelta del tipo di plot
    while True:
        plot_type = input(
            "\n📈 Che tipo di plot vuoi generare?\n   1) Solo 2D\n   2) Solo 3D\n   3) Entrambi\n   Scelta (1/2/3): "
        ).strip()

        if plot_type in ["1", "2", "3"]:
            break
        else:
            print("❌ Scelta non valida. Inserisci 1, 2 o 3.")

    # Carica i dati
    print(f"\n⏳ Caricamento dataset {stress_type}...")
    X, y, plant_ids = load_data(stress_type)

    if X is None:
        return

    # Normalizza i dati
    print("⏳ Normalizzazione dati con split standard...")
    X_normalized, y, mean_train, std_train = normalize_data_standard_split(X, y)

    # Prepara le frequenze
    frequencies = prepare_data(X_normalized)

    # Statistiche sui dati normalizzati
    print("\n📊 STATISTICHE DATI NORMALIZZATI:")
    print(
        f"   Range Real Part: [{np.min(X_normalized[:, :, 0]):.2f}, {np.max(X_normalized[:, :, 0]):.2f}]"
    )
    print(
        f"   Range Imag Part: [{np.min(X_normalized[:, :, 1]):.2f}, {np.max(X_normalized[:, :, 1]):.2f}]"
    )
    print(f"   Media Real Part: {np.mean(X_normalized[:, :, 0]):.4f} (≈ 0)")
    print(f"   Media Imag Part: {np.mean(X_normalized[:, :, 1]):.4f} (≈ 0)")
    print(f"   Std Real Part: {np.std(X_normalized[:, :, 0]):.4f} (≈ 1)")
    print(f"   Std Imag Part: {np.std(X_normalized[:, :, 1]):.4f} (≈ 1)")

    # Genera i plot richiesti
    if plot_type in ["1", "3"]:
        print("\n⏳ Generazione plot 2D...")
        fig_2d = plot_2d_normalized(X_normalized, y, frequencies, stress_type)
        fig_2d.show()
        print("✅ Plot 2D generato!")

    if plot_type in ["2", "3"]:
        print("\n⏳ Generazione plot 3D...")
        fig_3d = plot_3d_normalized(X_normalized, y, frequencies, stress_type)
        fig_3d.show()
        print("✅ Plot 3D generato!")

    print("\n" + "=" * 70)
    print("  VISUALIZZAZIONE COMPLETATA")
    print("=" * 70)
    print("  ℹ️  La normalizzazione Z-score rende i dati più comparabili")
    print("     rimuovendo le differenze di scala tra le features.")
    print("=" * 70)


if __name__ == "__main__":
    main()
