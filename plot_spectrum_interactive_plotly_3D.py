"""
plot_spectrum_interactive_plotly_3D.py

Script interattivo per visualizzare lo spettro di impedenza Real vs Imaginary
con filtri dinamici per piante, classi e range di frequenze.

Features:
- Visualizzazione interattiva con Plotly (2D e 3D)
- Filtri per piante (P0, P1, P3)
- Filtri per classi (Control, Early_Stress, Late_Stress)
- Selezione range di frequenze
- Opzione di normalizzazione Z-score
- Scelta tra visualizzazione 2D (Real vs Imaginary) o 3D (Real vs Imaginary vs Frequency)
- Salvataggio HTML in plots_plotly/

Author: Shanti Leonardo Arzu
Date: November 2025
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def load_dataset(stress_type):
    """
    Carica il dataset Water_Stress.npz o Iron_Stress.npz

    Args:
        stress_type (str): 'water' o 'iron'

    Returns:
        X (np.ndarray): Features (n_samples, 200, 2)
        y (np.ndarray): Labels (n_samples,)
        plant_ids (np.ndarray): Plant IDs (n_samples,)
    """
    # Path al dataset
    data_path = Path("./data")
    file_name = f"{stress_type.capitalize()}_Stress.npz"
    file_path = data_path / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {file_path}\n"
            f"Assicurati che il file sia nella cartella ./data/"
        )

    print(f"\n[INFO] Caricamento dataset: {file_path}")
    data = np.load(file_path, allow_pickle=True)

    X = data["X"]  # (n_samples, 400) flattened
    y = data["y"]  # (n_samples,)
    plant_ids = data["plant_ids"]  # (n_samples,)

    # Converti da flattened (n, 400) a shaped (n, 200, 2)
    X_shaped = X.reshape(-1, 200, 2)

    print("[INFO] Dataset caricato con successo!")
    print(f"      Shape: {X_shaped.shape}")
    print(f"      Piante uniche: {np.unique(plant_ids)}")
    print(f"      Classi uniche: {np.unique(y)}")

    return X_shaped, y, plant_ids


def generate_frequency_array():
    """
    Genera array logaritmico di frequenze da 100 Hz a 10 MHz (200 punti)

    Returns:
        freq_array (np.ndarray): Array di frequenze in Hz

    Note:
        Il dataset copre lo spettro da 100 Hz a 10 MHz distribuito
        logaritmicamente su 200 punti di frequenza.
    """
    return np.logspace(np.log10(100), np.log10(10e6), 200)


def normalize_data_zscore(X, y, plant_ids):
    """
    Normalizza i dati usando Z-score normalization.

    La normalizzazione viene calcolata per ogni frequenza e componente
    indipendentemente da piante e classi.

    Args:
        X (np.ndarray): Features (n_samples, 200, 2)
        y (np.ndarray): Labels (n_samples,)
        plant_ids (np.ndarray): Plant IDs (n_samples,)

    Returns:
        X_norm (np.ndarray): Features normalizzate (n_samples, 200, 2)
    """
    # Calcola mean e std su tutte le feature (tutti i campioni)
    mean = np.mean(X, axis=0, keepdims=True)  # (1, 200, 2)
    std = np.std(X, axis=0, keepdims=True)  # (1, 200, 2)

    # Protezione divisione per zero
    std[std == 0] = 1.0

    # Normalizza
    X_norm = (X - mean) / std

    print("\n[INFO] Normalizzazione Z-score applicata:")
    print(f"      Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"      Std range: [{std.min():.4f}, {std.max():.4f}]")
    print(f"      Normalized data - Mean: {X_norm.mean():.6f}, Std: {X_norm.std():.6f}")

    return X_norm


def filter_data(X, y, plant_ids, selected_plants, selected_classes, freq_range):
    """
    Filtra i dati in base a piante, classi e range di frequenze selezionati.

    Args:
        X (np.ndarray): Features (n_samples, 200, 2)
        y (np.ndarray): Labels (n_samples,)
        plant_ids (np.ndarray): Plant IDs (n_samples,)
        selected_plants (list): Liste di piante da includere, es. ['P0', 'P1']
        selected_classes (list): Liste di classi da includere, es. [0, 1, 2]
        freq_range (tuple): (freq_min, freq_max) in Hz

    Returns:
        X_filtered (np.ndarray): Features filtrate
        y_filtered (np.ndarray): Labels filtrate
        plant_ids_filtered (np.ndarray): Plant IDs filtrate
        freq_indices (np.ndarray): Indici delle frequenze selezionate
    """
    # Filtra per piante
    plant_mask = np.isin(plant_ids, selected_plants)

    # Filtra per classi
    class_mask = np.isin(y, selected_classes)

    # Combina i filtri
    combined_mask = plant_mask & class_mask

    X_filtered = X[combined_mask]
    y_filtered = y[combined_mask]
    plant_ids_filtered = plant_ids[combined_mask]

    # Filtra per range di frequenze
    freq_array = generate_frequency_array()
    freq_indices = np.where(
        (freq_array >= freq_range[0]) & (freq_array <= freq_range[1])
    )[0]

    X_filtered = X_filtered[:, freq_indices, :]

    print("\n[INFO] Dati filtrati:")
    print(f"      Piante selezionate: {selected_plants}")
    print(f"      Classi selezionate: {selected_classes}")
    print(f"      Range frequenze: {freq_range[0]:.0f} Hz - {freq_range[1]:.2e} Hz")
    print(f"      Campioni dopo filtro: {len(X_filtered)}")
    print(f"      Frequenze dopo filtro: {len(freq_indices)}")

    return X_filtered, y_filtered, plant_ids_filtered, freq_indices


def create_interactive_plot_2d(
    X, y, plant_ids, freq_indices, stress_type, normalized=False
):
    """
    Crea plot 2D interattivo (Real vs Imaginary).

    Args:
        X (np.ndarray): Features (n_samples, n_frequencies, 2)
        y (np.ndarray): Labels (n_samples,)
        plant_ids (np.ndarray): Plant IDs (n_samples,)
        freq_indices (np.ndarray): Indici delle frequenze visualizzate
        stress_type (str): 'water' o 'iron'
        normalized (bool): Se True, i dati sono normalizzati Z-score

    Returns:
        fig (plotly.graph_objects.Figure): Figura Plotly
    """
    # Mappatura classi → nomi e colori
    class_names = {0: "Control", 1: "Early_Stress", 2: "Late_Stress"}
    class_colors = {
        0: "#2ecc71",  # Verde
        1: "#f39c12",  # Arancione
        2: "#e74c3c",  # Rosso
    }

    # Frequenze selezionate
    freq_array = generate_frequency_array()[freq_indices]

    # Crea figura
    fig = go.Figure()

    # Itera su classi e piante per creare tracce separate
    unique_classes = np.unique(y)
    unique_plants = np.unique(plant_ids)

    for cls in unique_classes:
        for plant in unique_plants:
            # Filtra per classe e pianta
            mask = (y == cls) & (plant_ids == plant)
            X_subset = X[mask]

            if len(X_subset) == 0:
                continue

            # Estrai Real e Imaginary
            real_parts = X_subset[:, :, 0].flatten()
            imag_parts = X_subset[:, :, 1].flatten()

            # Crea hover text
            n_samples = X_subset.shape[0]
            freq_repeated = np.tile(freq_array, n_samples)

            hover_text = [
                f"Class: {class_names[cls]}<br>"
                f"Plant: {plant}<br>"
                f"Freq: {freq:.2e} Hz<br>"
                f"Real: {real:.2f}<br>"
                f"Imag: {imag:.2f}"
                for freq, real, imag in zip(freq_repeated, real_parts, imag_parts)
            ]

            # Aggiungi traccia
            fig.add_trace(
                go.Scattergl(
                    x=real_parts,
                    y=imag_parts,
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=class_colors[cls],
                        opacity=0.6,
                        line=dict(width=0),
                    ),
                    name=f"{class_names[cls]} - {plant}",
                    text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                    legendgroup=f"{class_names[cls]}",
                    showlegend=True,
                )
            )

    # Layout
    freq_min = freq_array.min()
    freq_max = freq_array.max()
    norm_suffix = " [Z-SCORE NORMALIZED]" if normalized else " [RAW DATA]"

    fig.update_layout(
        title=dict(
            text=f"2D Spectrum: Real vs Imaginary ({freq_min:.0f} Hz - {freq_max:.2e} Hz){norm_suffix}<br>"
            f"<sub>Stress Type: {stress_type.upper()}</sub>",
            x=0.5,
            xanchor="center",
            font=dict(size=20),
        ),
        xaxis=dict(
            title="Real Part",
            showgrid=True,
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="gray",
        ),
        yaxis=dict(
            title="Imaginary Part",
            showgrid=True,
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="gray",
        ),
        hovermode="closest",
        width=1200,
        height=800,
        template="plotly_white",
        legend=dict(
            title="Classes & Plants",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
    )

    return fig


def create_interactive_plot_3d(
    X, y, plant_ids, freq_indices, stress_type, normalized=False
):
    """
    Crea plot 3D interattivo (Real vs Imaginary vs Frequency).

    Args:
        X (np.ndarray): Features (n_samples, n_frequencies, 2)
        y (np.ndarray): Labels (n_samples,)
        plant_ids (np.ndarray): Plant IDs (n_samples,)
        freq_indices (np.ndarray): Indici delle frequenze visualizzate
        stress_type (str): 'water' o 'iron'
        normalized (bool): Se True, i dati sono normalizzati Z-score

    Returns:
        fig (plotly.graph_objects.Figure): Figura Plotly
    """
    # Mappatura classi → nomi e colori
    class_names = {0: "Control", 1: "Early_Stress", 2: "Late_Stress"}
    class_colors = {
        0: "#2ecc71",  # Verde
        1: "#f39c12",  # Arancione
        2: "#e74c3c",  # Rosso
    }

    # Frequenze selezionate
    freq_array = generate_frequency_array()[freq_indices]

    # Crea figura 3D
    fig = go.Figure()

    # Itera su classi e piante per creare tracce separate
    unique_classes = np.unique(y)
    unique_plants = np.unique(plant_ids)

    for cls in unique_classes:
        for plant in unique_plants:
            # Filtra per classe e pianta
            mask = (y == cls) & (plant_ids == plant)
            X_subset = X[mask]

            if len(X_subset) == 0:
                continue

            # Estrai Real, Imaginary e Frequency
            real_parts = X_subset[:, :, 0].flatten()
            imag_parts = X_subset[:, :, 1].flatten()

            # Crea array di frequenze per ogni punto
            n_samples = X_subset.shape[0]
            freq_repeated = np.tile(freq_array, n_samples)

            # Normalizza frequenze a scala logaritmica per visualizzazione
            freq_log = np.log10(freq_repeated)

            hover_text = [
                f"Class: {class_names[cls]}<br>"
                f"Plant: {plant}<br>"
                f"Freq: {freq:.2e} Hz<br>"
                f"Real: {real:.2f}<br>"
                f"Imag: {imag:.2f}"
                for freq, real, imag in zip(freq_repeated, real_parts, imag_parts)
            ]

            # Aggiungi traccia 3D
            fig.add_trace(
                go.Scatter3d(
                    x=real_parts,
                    y=imag_parts,
                    z=freq_log,  # Usa frequenza normalizzata come Z
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=class_colors[cls],
                        opacity=0.6,
                        line=dict(width=0),
                    ),
                    name=f"{class_names[cls]} - {plant}",
                    text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                    legendgroup=f"{class_names[cls]}",
                    showlegend=True,
                )
            )

    # Layout 3D
    freq_min = freq_array.min()
    freq_max = freq_array.max()
    norm_suffix = " [Z-SCORE NORMALIZED]" if normalized else " [RAW DATA]"

    fig.update_layout(
        title=dict(
            text=f"3D Spectrum: Real vs Imaginary vs Frequency ({freq_min:.0f} Hz - {freq_max:.2e} Hz){norm_suffix}<br>"
            f"<sub>Stress Type: {stress_type.upper()}</sub>",
            x=0.5,
            xanchor="center",
            font=dict(size=20),
        ),
        scene=dict(
            xaxis=dict(
                title="Real Part",
                showgrid=True,
                gridcolor="lightgray",
                zeroline=True,
                zerolinecolor="gray",
            ),
            yaxis=dict(
                title="Imaginary Part",
                showgrid=True,
                gridcolor="lightgray",
                zeroline=True,
                zerolinecolor="gray",
            ),
            zaxis=dict(
                title="Frequency (log10 scale)",
                showgrid=True,
                gridcolor="lightgray",
                zeroline=True,
                zerolinecolor="gray",
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)  # Angolo di visualizzazione iniziale
            ),
        ),
        hovermode="closest",
        width=1200,
        height=900,
        template="plotly_white",
        legend=dict(
            title="Classes & Plants",
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    return fig


def save_plot(
    fig,
    stress_type,
    selected_plants,
    selected_classes,
    freq_range,
    normalized=False,
    plot_type="2d",
):
    """
    Salva plot in formato HTML nella cartella plots_plotly/

    Args:
        fig (plotly.graph_objects.Figure): Figura Plotly
        stress_type (str): 'water' o 'iron'
        selected_plants (list): Piante selezionate
        selected_classes (list): Classi selezionate
        freq_range (tuple): Range di frequenze
        normalized (bool): Se True, i dati sono normalizzati
        plot_type (str): '2d' o '3d'
    """
    # Crea directory plots_plotly se non esiste
    output_dir = Path("./plots_plotly")
    output_dir.mkdir(exist_ok=True)

    # Genera nome file
    plants_str = "_".join(selected_plants)
    classes_str = "_".join([str(c) for c in selected_classes])
    freq_str = f"{freq_range[0]:.0f}Hz_{freq_range[1]:.2e}Hz"
    norm_str = "_normalized" if normalized else "_raw"
    type_str = f"_{plot_type}"

    filename = f"spectrum_{stress_type}_{plants_str}_classes{classes_str}_{freq_str}{norm_str}{type_str}.html"
    output_path = output_dir / filename

    # Salva HTML
    fig.write_html(str(output_path))

    print(f"\n[SUCCESS] Plot salvato in: {output_path}")
    print("[INFO] Apri il file con un browser per visualizzare il plot interattivo.")


def get_user_input():
    """
    Interfaccia utente per selezione parametri (senza argparse).

    Returns:
        config (dict): Configurazione selezionata dall'utente
    """
    print("=" * 80)
    print("VISUALIZZAZIONE INTERATTIVA SPETTRO IMPEDENZA")
    print("=" * 80)

    # Selezione stress type
    print("\n[1/6] Selezione tipo di stress:")
    print("  1. Water Stress")
    print("  2. Iron Stress")
    choice = input("Inserisci scelta (1 o 2): ").strip()
    stress_type = "water" if choice == "1" else "iron"
    print(f"✓ Selezionato: {stress_type.upper()} STRESS")

    # Selezione piante
    print("\n[2/6] Selezione piante da visualizzare:")
    print("  Opzioni: P0, P1, P3")
    print("  Esempi: 'P0' oppure 'P0,P1' oppure 'P0,P1,P3' (per tutte)")
    plants_input = input(
        "Inserisci piante separate da virgola (default: tutte): "
    ).strip()

    if plants_input == "":
        selected_plants = ["P0", "P1", "P3"]
    else:
        selected_plants = [p.strip().upper() for p in plants_input.split(",")]

    print(f"✓ Piante selezionate: {selected_plants}")

    # Selezione classi
    print("\n[3/6] Selezione classi da visualizzare:")
    print("  0 = Control (Verde)")
    print("  1 = Early Stress (Arancione)")
    print("  2 = Late Stress (Rosso)")
    print("  Esempi: '0' oppure '0,1' oppure '0,1,2' (per tutte)")
    classes_input = input(
        "Inserisci classi separate da virgola (default: tutte): "
    ).strip()

    if classes_input == "":
        selected_classes = [0, 1, 2]
    else:
        selected_classes = [int(c.strip()) for c in classes_input.split(",")]

    class_names = {0: "Control", 1: "Early", 2: "Late"}
    print(f"✓ Classi selezionate: {[class_names[c] for c in selected_classes]}")

    # Selezione range frequenze
    print("\n[4/6] Selezione range di frequenze:")
    print("  Range disponibile: 100 Hz - 10 MHz")
    print("  Default: tutto lo spettro (100 Hz - 10 MHz)")
    use_custom_range = (
        input("Vuoi specificare un range custom? (s/n, default: n): ").strip().lower()
    )

    if use_custom_range == "s":
        print("\n  [ATTENZIONE] Il range deve essere tra 100 Hz e 10 MHz")
        freq_min = float(input("  Frequenza minima (Hz, min=100): ").strip())
        freq_max = float(input("  Frequenza massima (Hz, max=10000000): ").strip())

        # Validazione range
        if freq_min < 100:
            print("  [WARNING] Frequenza minima < 100 Hz, impostata a 100 Hz")
            freq_min = 100
        if freq_max > 10e6:
            print("  [WARNING] Frequenza massima > 10 MHz, impostata a 10 MHz")
            freq_max = 10e6
        if freq_min >= freq_max:
            print("  [ERROR] Frequenza minima >= massima, uso range default")
            freq_range = (100, 10e6)
        else:
            freq_range = (freq_min, freq_max)

        print(f"✓ Range custom: {freq_range[0]:.0f} Hz - {freq_range[1]:.2e} Hz")
    else:
        freq_range = (100, 10e6)
        print("✓ Range default: 100 Hz - 10 MHz")

    # Selezione normalizzazione
    print("\n[5/6] Selezione normalizzazione dati:")
    print("  1. Raw Data (dati grezzi, non normalizzati)")
    print("  2. Z-Score Normalized (dati normalizzati)")
    norm_choice = input("Scegli tipo di visualizzazione (1 o 2, default: 1): ").strip()

    if norm_choice == "2":
        normalized = True
        print("✓ Visualizzazione: Z-Score Normalized")
    else:
        normalized = False
        print("✓ Visualizzazione: Raw Data")

    # Selezione tipo di plot
    print("\n[6/6] Selezione tipo di visualizzazione:")
    print("  1. 2D Plot (Real vs Imaginary)")
    print("  2. 3D Plot (Real vs Imaginary vs Frequency)")
    plot_choice = input("Scegli tipo di plot (1 o 2, default: 1): ").strip()

    if plot_choice == "2":
        plot_type = "3d"
        print("✓ Visualizzazione: 3D Plot")
    else:
        plot_type = "2d"
        print("✓ Visualizzazione: 2D Plot")

    return {
        "stress_type": stress_type,
        "selected_plants": selected_plants,
        "selected_classes": selected_classes,
        "freq_range": freq_range,
        "normalized": normalized,
        "plot_type": plot_type,
    }


def main():
    """
    Funzione principale che coordina il workflow.
    """
    # Ottieni input utente
    config = get_user_input()

    print("\n" + "=" * 80)
    print("ELABORAZIONE DATI")
    print("=" * 80)

    # Carica dataset
    X, y, plant_ids = load_dataset(config["stress_type"])

    # Applica normalizzazione se richiesta (PRIMA del filtering)
    if config["normalized"]:
        print("\n[INFO] Applicazione normalizzazione Z-score...")
        X = normalize_data_zscore(X, y, plant_ids)

    # Filtra dati
    X_filtered, y_filtered, plant_ids_filtered, freq_indices = filter_data(
        X,
        y,
        plant_ids,
        config["selected_plants"],
        config["selected_classes"],
        config["freq_range"],
    )

    # Crea plot interattivo in base a scelta tipo
    print("\n[INFO] Creazione plot interattivo...")

    if config["plot_type"] == "3d":
        fig = create_interactive_plot_3d(
            X_filtered,
            y_filtered,
            plant_ids_filtered,
            freq_indices,
            config["stress_type"],
            normalized=config["normalized"],
        )
    else:
        fig = create_interactive_plot_2d(
            X_filtered,
            y_filtered,
            plant_ids_filtered,
            freq_indices,
            config["stress_type"],
            normalized=config["normalized"],
        )

    # Salva plot
    save_plot(
        fig,
        config["stress_type"],
        config["selected_plants"],
        config["selected_classes"],
        config["freq_range"],
        normalized=config["normalized"],
        plot_type=config["plot_type"],
    )

    print("\n" + "=" * 80)
    print("✓ PROCESSO COMPLETATO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
