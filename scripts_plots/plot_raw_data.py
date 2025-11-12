"""
plot_raw_data.py

Script per visualizzare i dati di bioimpedenza delle piante NON NORMALIZZATI.
Genera grafici 2D e 3D usando Plotly per mostrare i dati RAW.

Author: Shanti Leonardo Arzu
Date: November 2025
"""

import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def prepare_data(X):
    """
    Prepara i dati per la visualizzazione
    Converte da (n_samples, 400) a (n_samples, 200, 2)
    """
    # Reshape da flattened a structured
    X_shaped = X.reshape(-1, 200, 2)  # (n_samples, n_frequencies, 2)

    # Frequenze logaritmiche da 100 Hz a 10 MHz
    frequencies = np.logspace(np.log10(100), np.log10(1e7), 200)

    return X_shaped, frequencies


def plot_2d_raw(X_shaped, y, frequencies, stress_type):
    """
    Crea plot 2D dei dati RAW
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
        class_data = X_shaped[mask]

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
        title=f"Trend di Frequenza (Full Spectrum) [RAW DATA]<br><sub>Stress Type: {stress_type.upper()}</sub>",
        height=800,
        showlegend=True,
        legend=dict(x=1.02, y=1),
        hovermode="x",
        template="plotly_white",
    )

    return fig


def plot_3d_raw(X_shaped, y, frequencies, stress_type):
    """
    Crea plot 3D dei dati RAW
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
        class_data = X_shaped[mask]

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
        title=f"3D Trend di Frequenza: Real vs Imaginary vs Frequency [RAW DATA]<br><sub>Stress Type: {stress_type.upper()}</sub>",
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
    print("  VISUALIZZAZIONE DATI RAW (NON NORMALIZZATI)")
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

    # Prepara i dati
    print("⏳ Preparazione dati...")
    X_shaped, frequencies = prepare_data(X)

    # Statistiche sui dati RAW
    print("\n📊 STATISTICHE DATI RAW:")
    print(
        f"   Range Real Part: [{np.min(X_shaped[:, :, 0]):.2f}, {np.max(X_shaped[:, :, 0]):.2f}]"
    )
    print(
        f"   Range Imag Part: [{np.min(X_shaped[:, :, 1]):.2f}, {np.max(X_shaped[:, :, 1]):.2f}]"
    )
    print(f"   Media Real Part: {np.mean(X_shaped[:, :, 0]):.2f}")
    print(f"   Media Imag Part: {np.mean(X_shaped[:, :, 1]):.2f}")
    print(f"   Std Real Part: {np.std(X_shaped[:, :, 0]):.2f}")
    print(f"   Std Imag Part: {np.std(X_shaped[:, :, 1]):.2f}")

    # Genera i plot richiesti
    if plot_type in ["1", "3"]:
        print("\n⏳ Generazione plot 2D...")
        fig_2d = plot_2d_raw(X_shaped, y, frequencies, stress_type)
        fig_2d.show()
        print("✅ Plot 2D generato!")

    if plot_type in ["2", "3"]:
        print("\n⏳ Generazione plot 3D...")
        fig_3d = plot_3d_raw(X_shaped, y, frequencies, stress_type)
        fig_3d.show()
        print("✅ Plot 3D generato!")

    print("\n" + "=" * 70)
    print("  VISUALIZZAZIONE COMPLETATA")
    print("=" * 70)


if __name__ == "__main__":
    main()
