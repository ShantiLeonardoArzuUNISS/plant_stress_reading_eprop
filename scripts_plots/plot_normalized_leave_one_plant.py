"""
plot_normalized_leave_one_plant.py

Script per visualizzare i dati di bioimpedenza delle piante NORMALIZZATI
con tecnica Leave-One-Plant-Out.
La normalizzazione usa media e std calcolati SOLO su 2 piante,
poi normalizza tutto il dataset (inclusa la pianta esclusa).

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


def normalize_data_leave_one_plant(X, y, plant_ids, leave_plant):
    """
    Normalizza i dati usando tecnica Leave-One-Plant-Out
    Calcola media e std SOLO sulle 2 piante di training
    """
    # Reshape da flattened a structured
    X_shaped = X.reshape(-1, 200, 2)  # (n_samples, n_frequencies, 2)

    # Maschere per separare le piante
    mask_leave_plant = plant_ids == leave_plant
    mask_train_plants = ~mask_leave_plant

    # Dati per training (2 piante)
    X_train = X_shaped[mask_train_plants]
    train_plants = plant_ids[mask_train_plants]

    print("\n📊 Leave-One-Plant-Out Split:")
    print(f"   Pianta esclusa: {leave_plant}")
    print(f"   Piante di training: {np.unique(train_plants)}")
    print(f"   Samples training (2 piante): {len(X_train)}")
    print(f"   Samples test (1 pianta): {np.sum(mask_leave_plant)}")

    # CALCOLA STATISTICHE SOLO SULLE 2 PIANTE DI TRAINING
    mean_train = np.mean(X_train, axis=0, keepdims=True)  # (1, 200, 2)
    std_train = np.std(X_train, axis=0, keepdims=True)  # (1, 200, 2)

    # Evita divisione per zero
    std_train[std_train == 0] = 1.0

    print(f"\n📊 Statistiche calcolate SOLO su piante {np.unique(train_plants)}:")
    print(f"   Mean range: [{mean_train.min():.4f}, {mean_train.max():.4f}]")
    print(f"   Std range: [{std_train.min():.4f}, {std_train.max():.4f}]")

    # NORMALIZZA TUTTO IL DATASET (inclusa la pianta esclusa)
    X_normalized = (X_shaped - mean_train) / std_train

    # Mostra distribuzione per pianta dopo normalizzazione
    print("\n📊 Dopo normalizzazione (con statistiche delle 2 piante di training):")
    for plant in np.unique(plant_ids):
        mask = plant_ids == plant
        mean_p = np.mean(X_normalized[mask])
        std_p = np.std(X_normalized[mask])
        status = (
            "(ESCLUSA dal calcolo statistiche)"
            if plant == leave_plant
            else "(training)"
        )
        print(f"   {plant} {status}: mean={mean_p:.4f}, std={std_p:.4f}")

    return X_normalized, y, plant_ids, mean_train, std_train


def prepare_data(X_normalized):
    """
    Prepara i dati normalizzati per la visualizzazione
    """
    # Frequenze logaritmiche da 100 Hz a 10 MHz
    frequencies = np.logspace(np.log10(100), np.log10(1e7), 200)
    return frequencies


def plot_2d_normalized_lopo(
    X_normalized, y, plant_ids, frequencies, stress_type, leave_plant
):
    """
    Crea plot 2D dei dati NORMALIZZATI con Leave-One-Plant-Out
    Mostra anche separazione tra piante
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

    # Trova piante di training
    train_plants = np.unique(plant_ids[plant_ids != leave_plant])

    # Update layout
    fig.update_layout(
        title=f"Trend di Frequenza [LEAVE-ONE-PLANT NORMALIZED]<br><sub>Stress: {stress_type.upper()} | Training: {train_plants} | Test: {leave_plant}</sub>",
        height=800,
        showlegend=True,
        legend=dict(x=1.02, y=1),
        hovermode="x",
        template="plotly_white",
    )

    return fig


def plot_3d_normalized_lopo(
    X_normalized, y, plant_ids, frequencies, stress_type, leave_plant
):
    """
    Crea plot 3D dei dati NORMALIZZATI con Leave-One-Plant-Out
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

    # Trova piante di training
    train_plants = np.unique(plant_ids[plant_ids != leave_plant])

    # Update layout
    fig.update_layout(
        title=f"3D Trend di Frequenza [LEAVE-ONE-PLANT NORMALIZED]<br><sub>Stress: {stress_type.upper()} | Training: {train_plants} | Test: {leave_plant}</sub>",
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
    print("  VISUALIZZAZIONE DATI NORMALIZZATI (LEAVE-ONE-PLANT-OUT)")
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

    # Carica i dati per vedere quali piante sono disponibili
    print(f"\n⏳ Caricamento dataset {stress_type}...")
    X, y, plant_ids = load_data(stress_type)

    if X is None:
        return

    # Scelta della pianta da escludere
    unique_plants = np.unique(plant_ids)
    print(f"\n🌱 Piante disponibili: {unique_plants}")

    while True:
        leave_plant = (
            input(
                f"\n🌱 Quale pianta vuoi escludere dal training?\n   Opzioni: {list(unique_plants)}\n   Scelta: "
            )
            .strip()
            .upper()
        )

        if leave_plant in unique_plants:
            break
        else:
            print(f"❌ Pianta non valida. Scegli tra {list(unique_plants)}")

    # Scelta del tipo di plot
    while True:
        plot_type = input(
            "\n📈 Che tipo di plot vuoi generare?\n   1) Solo 2D\n   2) Solo 3D\n   3) Entrambi\n   Scelta (1/2/3): "
        ).strip()

        if plot_type in ["1", "2", "3"]:
            break
        else:
            print("❌ Scelta non valida. Inserisci 1, 2 o 3.")

    # Normalizza i dati con Leave-One-Plant-Out
    print(f"\n⏳ Normalizzazione con Leave-One-Plant-Out (escludo {leave_plant})...")
    X_normalized, y, plant_ids, mean_train, std_train = normalize_data_leave_one_plant(
        X, y, plant_ids, leave_plant
    )

    # Prepara le frequenze
    frequencies = prepare_data(X_normalized)

    # Statistiche sui dati normalizzati
    print("\n📊 STATISTICHE DATI NORMALIZZATI (TUTTO IL DATASET):")
    print(
        f"   Range Real Part: [{np.min(X_normalized[:, :, 0]):.2f}, {np.max(X_normalized[:, :, 0]):.2f}]"
    )
    print(
        f"   Range Imag Part: [{np.min(X_normalized[:, :, 1]):.2f}, {np.max(X_normalized[:, :, 1]):.2f}]"
    )
    print(f"   Media Real Part: {np.mean(X_normalized[:, :, 0]):.4f}")
    print(f"   Media Imag Part: {np.mean(X_normalized[:, :, 1]):.4f}")
    print(f"   Std Real Part: {np.std(X_normalized[:, :, 0]):.4f}")
    print(f"   Std Imag Part: {np.std(X_normalized[:, :, 1]):.4f}")

    # Genera i plot richiesti
    if plot_type in ["1", "3"]:
        print("\n⏳ Generazione plot 2D...")
        fig_2d = plot_2d_normalized_lopo(
            X_normalized, y, plant_ids, frequencies, stress_type, leave_plant
        )
        fig_2d.show()
        print("✅ Plot 2D generato!")

    if plot_type in ["2", "3"]:
        print("\n⏳ Generazione plot 3D...")
        fig_3d = plot_3d_normalized_lopo(
            X_normalized, y, plant_ids, frequencies, stress_type, leave_plant
        )
        fig_3d.show()
        print("✅ Plot 3D generato!")

    print("\n" + "=" * 70)
    print("  VISUALIZZAZIONE COMPLETATA")
    print("=" * 70)
    print("  ℹ️  Leave-One-Plant-Out è utile per valutare la generalizzazione")
    print("     del modello su piante mai viste durante il training.")
    print(f"  ⚠️  La pianta {leave_plant} potrebbe mostrare distribuzioni diverse")
    print("     perché le statistiche sono calcolate solo sulle altre piante.")
    print("=" * 70)


if __name__ == "__main__":
    main()
