"""
plot_sergio_25_11_25.py

Script per visualizzare dati di bioimpedenza normalizzati con confronto
tra piante di training (Px + Py) e pianta esclusa (Pz).

Permette di:
- Scegliere dataset (Water/Iron Stress)
- Scegliere pianta da escludere
- Scegliere numero di samples per pianta
- Normalizzare usando mean/std delle piante non escluse
- Visualizzare grafici 2D e 3D interattivi

MODIFICHE: Colori distinti per Training vs Test, senza linee tratteggiate

Author: Shanti Leonardo Arzu
Date: 25 November 2025
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# FUNZIONI DI CARICAMENTO E NORMALIZZAZIONE
# =============================================================================


def load_data(stress_type):
    """
    Carica i dati dal file .npz

    Args:
        stress_type (str): 'water' o 'iron'

    Returns:
        X, y, plant_ids: Dati caricati
    """
    file_path = f"../data/{stress_type.capitalize()}_Stress.npz"

    if not os.path.exists(file_path):
        print(f"❌ File non trovato: {file_path}")
        print("   Assicurati che il file sia nella cartella corrente")
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


def normalize_with_custom_samples(
    X, y, plant_ids, leave_plant, n_samples_per_plant=224
):
    """
    Normalizza i dati usando un numero specifico di samples per pianta

    Args:
        X: Dati (n_samples, 400)
        y: Labels
        plant_ids: IDs piante
        leave_plant: Pianta da escludere
        n_samples_per_plant: Numero di samples da usare per pianta per calcolare statistiche

    Returns:
        X_normalized, y, plant_ids, mean_train, std_train, train_plants, excluded_plant_data
    """
    # Reshape da flattened a structured
    X_shaped = X.reshape(-1, 200, 2)  # (n_samples, n_frequencies, 2)

    # Identifica piante di training
    train_plants = np.unique(plant_ids[plant_ids != leave_plant])

    print("\n📊 Leave-One-Plant-Out Configuration:")
    print(f"   Pianta esclusa: {leave_plant}")
    print(f"   Piante di training: {train_plants}")

    # Raccogli i samples delle piante di training
    train_samples_list = []
    for plant in train_plants:
        plant_mask = plant_ids == plant
        plant_samples = X_shaped[plant_mask]

        # Limita al numero richiesto di samples
        if len(plant_samples) < n_samples_per_plant:
            print(
                f"   ⚠️  {plant}: solo {len(plant_samples)} samples disponibili (richiesti {n_samples_per_plant})"
            )
            selected_samples = plant_samples
        else:
            # Prendi i primi n_samples_per_plant
            selected_samples = plant_samples[:n_samples_per_plant]
            print(
                f"   ✓ {plant}: usando {n_samples_per_plant} samples di {len(plant_samples)} disponibili"
            )

        train_samples_list.append(selected_samples)

    # Unisci tutti i samples di training
    X_train = np.concatenate(train_samples_list, axis=0)

    print("\n📊 Samples utilizzati per calcolo statistiche:")
    print(f"   Totale samples training: {len(X_train)}")
    print(f"   Piante: {train_plants}")

    # CALCOLA STATISTICHE SOLO SUI SAMPLES SELEZIONATI
    mean_train = np.mean(X_train, axis=0, keepdims=True)  # (1, 200, 2)
    std_train = np.std(X_train, axis=0, keepdims=True)  # (1, 200, 2)

    # Evita divisione per zero
    std_train[std_train == 0] = 1.0

    print("\n📊 Statistiche calcolate:")
    print(f"   Mean range: [{mean_train.min():.4f}, {mean_train.max():.4f}]")
    print(f"   Std range: [{std_train.min():.4f}, {std_train.max():.4f}]")

    # NORMALIZZA TUTTO IL DATASET
    X_normalized = (X_shaped - mean_train) / std_train

    # Separa dati piante training e pianta esclusa
    train_mask = plant_ids != leave_plant
    excluded_mask = plant_ids == leave_plant

    train_data = {
        "X": X_normalized[train_mask],
        "y": y[train_mask],
        "plant_ids": plant_ids[train_mask],
    }

    excluded_data = {
        "X": X_normalized[excluded_mask],
        "y": y[excluded_mask],
        "plant_ids": plant_ids[excluded_mask],
    }

    # Mostra statistiche dopo normalizzazione
    print("\n📊 Dopo normalizzazione:")
    for plant in np.unique(plant_ids):
        mask = plant_ids == plant
        mean_p = np.mean(X_normalized[mask])
        std_p = np.std(X_normalized[mask])
        status = "(ESCLUSA)" if plant == leave_plant else "(training)"
        print(f"   {plant} {status}: mean={mean_p:.4f}, std={std_p:.4f}")

    return X_normalized, y, plant_ids, mean_train, std_train, train_data, excluded_data


# =============================================================================
# FUNZIONI DI PLOTTING
# =============================================================================


def plot_2d_comparison(
    train_data,
    excluded_data,
    frequencies,
    stress_type,
    leave_plant,
    train_plants,
    vis_mode,
):
    """
    Crea plot 2D che confronta piante training vs pianta esclusa

    Args:
        train_data: Dizionario con dati training
        excluded_data: Dizionario con dati pianta esclusa
        frequencies: Array frequenze
        stress_type: Tipo di stress
        leave_plant: Pianta esclusa
        train_plants: Piante di training
        vis_mode: 1=medie, 2=individuali, 3=entrambe
    """
    # COLORI TRAINING (Px + Py) - Tonalità calde/neutre
    colors_train = {
        0: "#2ecc71",  # Verde brillante - Control
        1: "#f39c12",  # Arancione - Early
        2: "#e74c3c",  # Rosso - Late
    }

    # COLORI TEST (Pz) - Tonalità fredde/viola
    colors_test = {
        0: "#3498db",  # Blu - Control
        1: "#9b59b6",  # Viola - Early
        2: "#e91e63",  # Magenta - Late
    }

    class_names = {0: "Control", 1: "Early_Stress", 2: "Late_Stress"}

    # Crea subplot
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"Parte Reale - Training {train_plants} vs Test {leave_plant}",
            f"Parte Immaginaria - Training {train_plants} vs Test {leave_plant}",
        ),
        vertical_spacing=0.15,
    )

    # Per ogni classe
    for cls in sorted(np.unique(train_data["y"])):
        # Dati training per questa classe
        train_mask = train_data["y"] == cls
        train_class_data = train_data["X"][train_mask]

        # Dati pianta esclusa per questa classe
        excl_mask = excluded_data["y"] == cls
        excl_class_data = excluded_data["X"][excl_mask]

        if len(train_class_data) == 0 or len(excl_class_data) == 0:
            continue

        # --- PLOT SAMPLES INDIVIDUALI (se richiesto) ---
        if vis_mode in [2, 3]:
            # Training samples
            for i in range(min(5, len(train_class_data))):  # Max 5 samples
                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=train_class_data[i, :, 0],
                        mode="lines",
                        name=f"{class_names[cls]} Train",
                        line=dict(color=colors_train[cls], width=1),
                        opacity=0.3,
                        legendgroup=f"{class_names[cls]}_train",
                        showlegend=(i == 0),
                    ),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=train_class_data[i, :, 1],
                        mode="lines",
                        name=f"{class_names[cls]} Train",
                        line=dict(color=colors_train[cls], width=1),
                        opacity=0.3,
                        legendgroup=f"{class_names[cls]}_train",
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

            # Excluded samples - COLORI DIVERSI, LINEE SOLIDE
            for i in range(min(5, len(excl_class_data))):
                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=excl_class_data[i, :, 0],
                        mode="lines",
                        name=f"{class_names[cls]} Test ({leave_plant})",
                        line=dict(color=colors_test[cls], width=1),  # Colore diverso, nessun dash
                        opacity=0.5,
                        legendgroup=f"{class_names[cls]}_test",
                        showlegend=(i == 0),
                    ),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=frequencies,
                        y=excl_class_data[i, :, 1],
                        mode="lines",
                        name=f"{class_names[cls]} Test ({leave_plant})",
                        line=dict(color=colors_test[cls], width=1),  # Colore diverso, nessun dash
                        opacity=0.5,
                        legendgroup=f"{class_names[cls]}_test",
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

        # --- PLOT MEDIE (se richiesto) ---
        if vis_mode in [1, 3]:
            # Medie training
            train_real_mean = np.mean(train_class_data[:, :, 0], axis=0)
            train_real_std = np.std(train_class_data[:, :, 0], axis=0)
            train_imag_mean = np.mean(train_class_data[:, :, 1], axis=0)
            train_imag_std = np.std(train_class_data[:, :, 1], axis=0)

            # Medie pianta esclusa
            excl_real_mean = np.mean(excl_class_data[:, :, 0], axis=0)
            excl_real_std = np.std(excl_class_data[:, :, 0], axis=0)
            excl_imag_mean = np.mean(excl_class_data[:, :, 1], axis=0)
            excl_imag_std = np.std(excl_class_data[:, :, 1], axis=0)

            # Plot Real Part - Training Mean
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=train_real_mean,
                    mode="lines",
                    name=f"{class_names[cls]} Train (media)",
                    line=dict(color=colors_train[cls], width=3),
                    legendgroup=f"{class_names[cls]}_train_mean",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            # Banda confidenza training - Real
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([frequencies, frequencies[::-1]]),
                    y=np.concatenate(
                        [
                            train_real_mean + train_real_std,
                            (train_real_mean - train_real_std)[::-1],
                        ]
                    ),
                    fill="toself",
                    fillcolor=colors_train[cls],
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    legendgroup=f"{class_names[cls]}_train_mean",
                    opacity=0.2,
                ),
                row=1,
                col=1,
            )

            # Plot Real Part - Excluded Mean - COLORE DIVERSO, LINEA SOLIDA
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=excl_real_mean,
                    mode="lines",
                    name=f"{class_names[cls]} Test {leave_plant} (media)",
                    line=dict(color=colors_test[cls], width=3),  # Colore diverso, nessun dash
                    legendgroup=f"{class_names[cls]}_test_mean",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            # Banda confidenza excluded - Real
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([frequencies, frequencies[::-1]]),
                    y=np.concatenate(
                        [
                            excl_real_mean + excl_real_std,
                            (excl_real_mean - excl_real_std)[::-1],
                        ]
                    ),
                    fill="toself",
                    fillcolor=colors_test[cls],
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    legendgroup=f"{class_names[cls]}_test_mean",
                    opacity=0.15,
                ),
                row=1,
                col=1,
            )

            # Plot Imaginary Part - Training Mean
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=train_imag_mean,
                    mode="lines",
                    name=f"{class_names[cls]} Train (media)",
                    line=dict(color=colors_train[cls], width=3),
                    legendgroup=f"{class_names[cls]}_train_mean",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            # Banda confidenza training - Imaginary
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([frequencies, frequencies[::-1]]),
                    y=np.concatenate(
                        [
                            train_imag_mean + train_imag_std,
                            (train_imag_mean - train_imag_std)[::-1],
                        ]
                    ),
                    fill="toself",
                    fillcolor=colors_train[cls],
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    legendgroup=f"{class_names[cls]}_train_mean",
                    opacity=0.2,
                ),
                row=2,
                col=1,
            )

            # Plot Imaginary Part - Excluded Mean - COLORE DIVERSO, LINEA SOLIDA
            fig.add_trace(
                go.Scatter(
                    x=frequencies,
                    y=excl_imag_mean,
                    mode="lines",
                    name=f"{class_names[cls]} Test {leave_plant} (media)",
                    line=dict(color=colors_test[cls], width=3),  # Colore diverso, nessun dash
                    legendgroup=f"{class_names[cls]}_test_mean",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            # Banda confidenza excluded - Imaginary
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([frequencies, frequencies[::-1]]),
                    y=np.concatenate(
                        [
                            excl_imag_mean + excl_imag_std,
                            (excl_imag_mean - excl_imag_std)[::-1],
                        ]
                    ),
                    fill="toself",
                    fillcolor=colors_test[cls],
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    legendgroup=f"{class_names[cls]}_test_mean",
                    opacity=0.15,
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
        title_text="Parte Reale (Normalizzata)", row=1, col=1, gridcolor="lightgray"
    )
    fig.update_yaxes(
        title_text="Parte Immaginaria (Normalizzata)",
        row=2,
        col=1,
        gridcolor="lightgray",
    )

    # Update layout
    vis_mode_text = {1: "Medie", 2: "Individuali", 3: "Medie + Individuali"}
    fig.update_layout(
        title=f"Confronto 2D [{vis_mode_text[vis_mode]}] - Stress: {stress_type.upper()}<br>"
        + f"Training: {train_plants} vs Test: {leave_plant}",
        height=900,
        showlegend=True,
        legend=dict(x=1.02, y=1),
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def plot_3d_comparison(
    train_data,
    excluded_data,
    frequencies,
    stress_type,
    leave_plant,
    train_plants,
    vis_mode,
):
    """
    Crea plot 3D che confronta piante training vs pianta esclusa
    """
    # COLORI TRAINING (Px + Py) - Tonalità calde/neutre
    colors_train = {
        0: "#2ecc71",  # Verde brillante
        1: "#f39c12",  # Arancione
        2: "#e74c3c",  # Rosso
    }

    # COLORI TEST (Pz) - Tonalità fredde/viola
    colors_test = {
        0: "#3498db",  # Blu
        1: "#9b59b6",  # Viola
        2: "#e91e63",  # Magenta
    }

    class_names = {0: "Control", 1: "Early_Stress", 2: "Late_Stress"}

    fig = go.Figure()

    # Frequenze in scala logaritmica
    freq_log = np.log10(frequencies)

    for cls in sorted(np.unique(train_data["y"])):
        train_mask = train_data["y"] == cls
        train_class_data = train_data["X"][train_mask]

        excl_mask = excluded_data["y"] == cls
        excl_class_data = excluded_data["X"][excl_mask]

        if len(train_class_data) == 0 or len(excl_class_data) == 0:
            continue

        # --- PLOT SAMPLES INDIVIDUALI (se richiesto) ---
        if vis_mode in [2, 3]:
            for i in range(min(3, len(train_class_data))):
                fig.add_trace(
                    go.Scatter3d(
                        x=train_class_data[i, :, 0],
                        y=train_class_data[i, :, 1],
                        z=freq_log,
                        mode="lines",
                        name=f"{class_names[cls]} Train",
                        line=dict(color=colors_train[cls], width=3),
                        opacity=0.4,
                        legendgroup=f"{class_names[cls]}_train",
                        showlegend=(i == 0),
                    )
                )

            # Excluded samples - COLORE DIVERSO, LINEA SOLIDA
            for i in range(min(3, len(excl_class_data))):
                fig.add_trace(
                    go.Scatter3d(
                        x=excl_class_data[i, :, 0],
                        y=excl_class_data[i, :, 1],
                        z=freq_log,
                        mode="lines",
                        name=f"{class_names[cls]} Test ({leave_plant})",
                        line=dict(color=colors_test[cls], width=3),  # Colore diverso, nessun dash
                        opacity=0.6,
                        legendgroup=f"{class_names[cls]}_test",
                        showlegend=(i == 0),
                    )
                )

        # --- PLOT MEDIE (se richiesto) ---
        if vis_mode in [1, 3]:
            train_real_mean = np.mean(train_class_data[:, :, 0], axis=0)
            train_imag_mean = np.mean(train_class_data[:, :, 1], axis=0)

            excl_real_mean = np.mean(excl_class_data[:, :, 0], axis=0)
            excl_imag_mean = np.mean(excl_class_data[:, :, 1], axis=0)

            # Training mean
            fig.add_trace(
                go.Scatter3d(
                    x=train_real_mean,
                    y=train_imag_mean,
                    z=freq_log,
                    mode="lines+markers",
                    name=f"{class_names[cls]} Train (media)",
                    line=dict(color=colors_train[cls], width=5),
                    marker=dict(size=4, color=colors_train[cls]),
                    legendgroup=f"{class_names[cls]}_train_mean",
                    showlegend=True,
                )
            )

            # Excluded mean - COLORE DIVERSO, LINEA SOLIDA, MARKER DIVERSO
            fig.add_trace(
                go.Scatter3d(
                    x=excl_real_mean,
                    y=excl_imag_mean,
                    z=freq_log,
                    mode="lines+markers",
                    name=f"{class_names[cls]} Test {leave_plant} (media)",
                    line=dict(color=colors_test[cls], width=5),  # Colore diverso, nessun dash
                    marker=dict(size=5, color=colors_test[cls], symbol="diamond"),
                    legendgroup=f"{class_names[cls]}_test_mean",
                    showlegend=True,
                )
            )

    vis_mode_text = {1: "Medie", 2: "Individuali", 3: "Medie + Individuali"}
    fig.update_layout(
        title=f"Confronto 3D [{vis_mode_text[vis_mode]}] - Stress: {stress_type.upper()}<br>"
        + f"Training: {train_plants} vs Test: {leave_plant}",
        scene=dict(
            xaxis_title="Parte Reale (Normalizzata)",
            yaxis_title="Parte Immaginaria (Normalizzata)",
            zaxis_title="Frequenza (log10 Hz)",
            xaxis=dict(gridcolor="lightgray"),
            yaxis=dict(gridcolor="lightgray"),
            zaxis=dict(gridcolor="lightgray"),
        ),
        height=800,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )

    return fig


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Funzione principale con interfaccia utente interattiva
    """
    print("\n" + "=" * 70)
    print("   PLOT SERGIO 25/11/25")
    print("   Confronto Training Plants vs Excluded Plant (Normalizzato)")
    print("=" * 70)

    # --- SCELTA DATASET ---
    while True:
        stress_type = input(
            "\n📊 Quale dataset vuoi visualizzare?\n"
            + "   1) Water Stress\n"
            + "   2) Iron Stress\n"
            + "   Scelta (1/2): "
        ).strip()

        if stress_type == "1":
            stress_type = "water"
            break
        elif stress_type == "2":
            stress_type = "iron"
            break
        else:
            print("❌ Scelta non valida. Inserisci 1 o 2.")

    # --- CARICA DATI ---
    print(f"\n⏳ Caricamento dataset {stress_type}...")
    X, y, plant_ids = load_data(stress_type)

    if X is None:
        return

    # --- SCELTA PIANTA DA ESCLUDERE ---
    unique_plants = np.unique(plant_ids)
    print(f"\n🌱 Piante disponibili: {unique_plants}")

    while True:
        leave_plant = (
            input(
                "\n🌱 Quale pianta vuoi escludere dal training?\n"
                + f"   Opzioni: {list(unique_plants)}\n"
                + "   Scelta: "
            )
            .strip()
            .upper()
        )

        if leave_plant in unique_plants:
            break
        else:
            print(f"❌ Pianta non valida. Scegli tra {list(unique_plants)}")

    # --- SCELTA NUMERO SAMPLES ---
    # Calcola max samples disponibile per pianta
    max_samples = min(
        [np.sum(plant_ids == p) for p in unique_plants if p != leave_plant]
    )

    while True:
        n_samples_input = input(
            "\n📊 Quanti samples utilizzare per pianta per calcolare mean/std?\n"
            + f"   Range: 1-{max_samples} (default: {max_samples})\n"
            + "   Scelta: "
        ).strip()

        if n_samples_input == "":
            n_samples_per_plant = max_samples
            break

        try:
            n_samples_per_plant = int(n_samples_input)
            if 1 <= n_samples_per_plant <= max_samples:
                break
            else:
                print(
                    f"❌ Valore fuori range. Inserisci un numero tra 1 e {max_samples}."
                )
        except ValueError:
            print("❌ Inserisci un numero valido.")

    # --- NORMALIZZAZIONE ---
    print(f"\n⏳ Normalizzazione con {n_samples_per_plant} samples per pianta...")
    X_normalized, y, plant_ids, mean_train, std_train, train_data, excluded_data = (
        normalize_with_custom_samples(X, y, plant_ids, leave_plant, n_samples_per_plant)
    )

    train_plants = np.unique(train_data["plant_ids"])

    # --- SCELTA TIPO PLOT ---
    while True:
        plot_type = input(
            "\n📈 Che tipo di plot vuoi generare?\n"
            + "   1) Solo 2D\n"
            + "   2) Solo 3D\n"
            + "   3) Entrambi\n"
            + "   Scelta (1/2/3): "
        ).strip()

        if plot_type in ["1", "2", "3"]:
            plot_type = int(plot_type)
            break
        else:
            print("❌ Scelta non valida. Inserisci 1, 2 o 3.")

    # --- SCELTA MODALITÀ VISUALIZZAZIONE ---
    while True:
        vis_mode = input(
            "\n👁️  Modalità di visualizzazione:\n"
            + "   1) Solo medie (con banda di confidenza)\n"
            + "   2) Solo letture individuali (samples singoli)\n"
            + "   3) Entrambe (individuali + medie)\n"
            + "   Scelta (1/2/3): "
        ).strip()

        if vis_mode in ["1", "2", "3"]:
            vis_mode = int(vis_mode)
            break
        else:
            print("❌ Scelta non valida. Inserisci 1, 2 o 3.")

    # --- PREPARA FREQUENZE ---
    frequencies = np.logspace(np.log10(100), np.log10(1e7), 200)

    # --- GENERA PLOT ---
    if plot_type in [1, 3]:
        print("\n⏳ Generazione plot 2D...")
        fig_2d = plot_2d_comparison(
            train_data,
            excluded_data,
            frequencies,
            stress_type,
            leave_plant,
            train_plants,
            vis_mode,
        )
        fig_2d.show()
        print("✅ Plot 2D generato!")

    if plot_type in [2, 3]:
        print("\n⏳ Generazione plot 3D...")
        fig_3d = plot_3d_comparison(
            train_data,
            excluded_data,
            frequencies,
            stress_type,
            leave_plant,
            train_plants,
            vis_mode,
        )
        fig_3d.show()
        print("✅ Plot 3D generato!")

    # --- RIEPILOGO FINALE ---
    print("\n" + "=" * 70)
    print("   ✅ VISUALIZZAZIONE COMPLETATA")
    print("=" * 70)
    print(f"   📊 Dataset: {stress_type.upper()} Stress")
    print(f"   🌱 Training plants: {train_plants}")
    print(f"   🌱 Test plant: {leave_plant}")
    print(f"   📈 Samples per pianta (per statistiche): {n_samples_per_plant}")
    print(f"   📊 Normalizzazione: mean/std calcolati su {train_plants}")
    print(
        f"   🎨 Plot generati: {'2D' if plot_type == 1 else '3D' if plot_type == 2 else '2D + 3D'}"
    )
    print(
        f"   👁️  Modalità: {'Medie' if vis_mode == 1 else 'Individuali' if vis_mode == 2 else 'Medie + Individuali'}"
    )
    print("=" * 70)
    print("\n   ℹ️  I grafici mostrano la differenza tra:")
    print(
        f"      - Piante di training {train_plants} (tonalità calde: verde/arancione/rosso)"
    )
    print(
        f"      - Pianta esclusa {leave_plant} (tonalità fredde: blu/viola/magenta)"
    )
    print("   🎨 Colori distinti senza linee tratteggiate per migliore visibilità")
    print("=" * 70)


if __name__ == "__main__":
    main()
