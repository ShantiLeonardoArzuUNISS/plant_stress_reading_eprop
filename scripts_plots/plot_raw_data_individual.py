"""
plot_raw_data_individual.py

Script per visualizzare i dati di bioimpedenza delle piante NON NORMALIZZATI.
Genera grafici 2D e 3D usando Plotly con opzione per visualizzare:
- Solo medie per classe (con banda di confidenza)
- Tutte le letture singolarmente
- Entrambe le visualizzazioni

Author: Shanti Leonardo Arzu
Date: November 2025
Modified: November 2025 - Aggiunta visualizzazione letture individuali
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


def plot_2d_individual(X_shaped, y, plant_ids, frequencies, stress_type, show_mode="both"):
    """
    Crea plot 2D dei dati RAW con opzione per visualizzare letture individuali

    Parameters:
    -----------
    show_mode : str
        - "mean": mostra solo le medie con banda di confidenza
        - "individual": mostra tutte le letture singolarmente
        - "both": mostra sia le letture individuali che le medie
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

        # Se mostra letture individuali
        if show_mode in ["individual", "both"]:
            # Mostra ogni singola lettura
            for idx in range(len(class_data)):
                show_legend = idx == 0  # Solo la prima traccia nella legenda

                # Plot Real Part - lettura singola
                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=class_data[idx, :, 0],
                        mode="lines",
                        name=f"{class_names[cls]} (singola)" if show_legend else None,
                        line=dict(color=colors[cls], width=0.5),
                        opacity=0.3,
                        legendgroup=f"{class_names[cls]}_individual",
                        showlegend=show_legend,
                        hovertemplate=f"Classe: {class_names[cls]}<br>Freq: %{{x:.2f}} Hz<br>Real: %{{y:.2f}}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

                # Plot Imaginary Part - lettura singola
                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=class_data[idx, :, 1],
                        mode="lines",
                        name=None,
                        line=dict(color=colors[cls], width=0.5),
                        opacity=0.3,
                        legendgroup=f"{class_names[cls]}_individual",
                        showlegend=False,
                        hovertemplate=f"Classe: {class_names[cls]}<br>Freq: %{{x:.2f}} Hz<br>Imag: %{{y:.2f}}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

        # Se mostra medie
        if show_mode in ["mean", "both"]:
            # Calcola media e deviazione standard
            real_mean = np.mean(class_data[:, :, 0], axis=0)
            real_std = np.std(class_data[:, :, 0], axis=0)

            imag_mean = np.mean(class_data[:, :, 1], axis=0)
            imag_std = np.std(class_data[:, :, 1], axis=0)

            # Plot Real Part - Media
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=real_mean,
                    mode="lines",
                    name=f"{class_names[cls]} (media)",
                    line=dict(color=colors[cls], width=3),
                    legendgroup=f"{class_names[cls]}_mean",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            # Aggiungi banda di confidenza (std) per Real
            if show_mode == "mean" or show_mode == "both":
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([frequencies, frequencies[::-1]]),
                        y=np.concatenate([real_mean + real_std, (real_mean - real_std)[::-1]]),
                        fill="toself",
                        fillcolor=colors[cls],
                        line=dict(color="rgba(255,255,255,0)"),
                        showlegend=False,
                        legendgroup=f"{class_names[cls]}_mean",
                        opacity=0.2,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )

            # Plot Imaginary Part - Media
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=imag_mean,
                    mode="lines",
                    name=None,
                    line=dict(color=colors[cls], width=3),
                    legendgroup=f"{class_names[cls]}_mean",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            # Aggiungi banda di confidenza (std) per Imaginary
            if show_mode == "mean" or show_mode == "both":
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([frequencies, frequencies[::-1]]),
                        y=np.concatenate([imag_mean + imag_std, (imag_mean - imag_std)[::-1]]),
                        fill="toself",
                        fillcolor=colors[cls],
                        line=dict(color="rgba(255,255,255,0)"),
                        showlegend=False,
                        legendgroup=f"{class_names[cls]}_mean",
                        opacity=0.2,
                        hoverinfo="skip",
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

    y_label_real = "Real Part"
    y_label_imag = "Imaginary Part"
    if show_mode == "mean":
        y_label_real += " (Media ± Std)"
        y_label_imag += " (Media ± Std)"
    elif show_mode == "individual":
        y_label_real += " (Letture Singole)"
        y_label_imag += " (Letture Singole)"
    else:
        y_label_real += " (Singole + Media)"
        y_label_imag += " (Singole + Media)"

    fig.update_yaxes(title_text=y_label_real, row=1, col=1, gridcolor="lightgray")
    fig.update_yaxes(title_text=y_label_imag, row=2, col=1, gridcolor="lightgray")

    # Update layout
    title_suffix = {
        "mean": " - Solo Medie",
        "individual": " - Letture Individuali",
        "both": " - Individuali + Medie"
    }

    fig.update_layout(
        title=f"Trend di Frequenza (Full Spectrum) [RAW DATA]{title_suffix[show_mode]}<br><sub>Stress Type: {stress_type.upper()}</sub>",
        height=800,
        showlegend=True,
        legend=dict(x=1.02, y=1),
        hovermode="closest",
        template="plotly_white",
    )

    return fig


def plot_3d_individual(X_shaped, y, plant_ids, frequencies, stress_type, show_mode="both"):
    """
    Crea plot 3D dei dati RAW con opzione per visualizzare letture individuali

    Parameters:
    -----------
    show_mode : str
        - "mean": mostra solo le medie
        - "individual": mostra tutte le letture singolarmente
        - "both": mostra sia le letture individuali che le medie
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

        # Frequenze in scala logaritmica per visualizzazione
        freq_log = np.log10(frequencies)

        # Se mostra letture individuali
        if show_mode in ["individual", "both"]:
            # Mostra ogni singola lettura
            for idx in range(len(class_data)):
                show_legend = idx == 0  # Solo la prima traccia nella legenda

                real_values = class_data[idx, :, 0]
                imag_values = class_data[idx, :, 1]

                fig.add_trace(
                    go.Scatter3d(
                        x=real_values,
                        y=imag_values,
                        z=freq_log,
                        mode="lines",
                        name=f"{class_names[cls]} (singola)" if show_legend else None,
                        line=dict(color=colors[cls], width=2),
                        opacity=0.3,
                        legendgroup=f"{class_names[cls]}_individual",
                        showlegend=show_legend,
                        hovertemplate=f"Classe: {class_names[cls]}<br>Real: %{{x:.2f}}<br>Imag: %{{y:.2f}}<br>Freq: %{{z:.2f}}<extra></extra>",
                    )
                )

        # Se mostra medie
        if show_mode in ["mean", "both"]:
            # Calcola media per ogni frequenza
            real_mean = np.mean(class_data[:, :, 0], axis=0)
            imag_mean = np.mean(class_data[:, :, 1], axis=0)

            fig.add_trace(
                go.Scatter3d(
                    x=real_mean,
                    y=imag_mean,
                    z=freq_log,
                    mode="lines+markers",
                    name=f"{class_names[cls]} (media)",
                    line=dict(color=colors[cls], width=6),
                    marker=dict(size=4, color=colors[cls]),
                    legendgroup=f"{class_names[cls]}_mean",
                    showlegend=True,
                )
            )

    # Update layout
    title_suffix = {
        "mean": " - Solo Medie",
        "individual": " - Letture Individuali",
        "both": " - Individuali + Medie"
    }

    fig.update_layout(
        title=f"3D Trend: Real vs Imaginary vs Frequency [RAW DATA]{title_suffix[show_mode]}<br><sub>Stress Type: {stress_type.upper()}</sub>",
        scene=dict(
            xaxis_title="Real Part",
            yaxis_title="Imaginary Part",
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
    print("  Versione con Letture Individuali")
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

    # Scelta della modalità di visualizzazione
    while True:
        show_mode = input(
            "\n👁️  Modalità di visualizzazione:\n   1) Solo medie (con banda di confidenza)\n   2) Solo letture individuali\n   3) Entrambe (individuali + medie)\n   Scelta (1/2/3): "
        ).strip()

        if show_mode == "1":
            show_mode = "mean"
            break
        elif show_mode == "2":
            show_mode = "individual"
            break
        elif show_mode == "3":
            show_mode = "both"
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
    print(f"   Numero totale di letture: {len(X_shaped)}")

    # Genera i plot richiesti
    if plot_type in ["1", "3"]:
        print("\n⏳ Generazione plot 2D...")
        fig_2d = plot_2d_individual(X_shaped, y, plant_ids, frequencies, stress_type, show_mode)
        fig_2d.show()
        print("✅ Plot 2D generato!")

    if plot_type in ["2", "3"]:
        print("\n⏳ Generazione plot 3D...")
        fig_3d = plot_3d_individual(X_shaped, y, plant_ids, frequencies, stress_type, show_mode)
        fig_3d.show()
        print("✅ Plot 3D generato!")

    print("\n" + "=" * 70)
    print("  VISUALIZZAZIONE COMPLETATA")
    print("=" * 70)


if __name__ == "__main__":
    main()
