"""
Script analisi dataset piante con normalizzazione Z-score
Genera 3 gruppi di grafici per analizzare stress nelle piante
Include sezione di confronto dati normalizzati vs non normalizzati
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# ======================== CONFIGURAZIONE ========================
STRESS_TYPE = "water"  # Cambia in "iron" per l'altro dataset
GENERATE_INDIVIDUAL = False  # True per generare 2016 grafici singoli
GENERATE_NORMALIZATION_COMPARISON = True  # True per generare confronto normalizzazione

# Parametri in base al tipo di stress
if STRESS_TYPE == "water":
    OPTIMAL_FREQ_IDX = [0, 1, 2]  # Frequenze 100-120 Hz
    FREQ_RANGE = "100-120 Hz"
else:
    OPTIMAL_FREQ_IDX = [197, 198, 199]  # Frequenze 4.7-10 MHz
    FREQ_RANGE = "4.7-10 MHz"

# Nomi e colori delle classi
CLASS_NAMES = {0: "Control", 1: "Early_Stress", 2: "Late_Stress"}
CLASS_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

# ======================== CONFIGURAZIONE RISOLUZIONE ========================
DPI_GROUPED = 200
DPI_OPTIMAL = 200
DPI_FULL_SPECTRUM = 200
DPI_INDIVIDUAL = 150
DPI_NORMALIZATION = 200  # Per grafici normalizzazione

FIGSIZE_GROUPED = (20, 18)
FIGSIZE_OPTIMAL_COMP = (22, 10)
FIGSIZE_OPTIMAL_TRENDS = (18, 14)
FIGSIZE_FULL_SPEC_COMP = (22, 10)
FIGSIZE_FULL_SPEC_TRENDS = (20, 14)
FIGSIZE_INDIVIDUAL = (12, 10)
FIGSIZE_NORMALIZATION = (24, 14)  # Per confronto normalizzazione

# ======================== FUNZIONI BASE ========================


def load_dataset(stress_type):
    """Carica il dataset NPZ"""
    file_path = f"./data/{stress_type.capitalize()}_Stress.npz"
    data = np.load(file_path, allow_pickle=True)

    X = data["X"]  # Features (n_samples, 400)
    y = data["y"]  # Labels
    plant_ids = data["plant_ids"]  # ID piante

    print(f"\n=== CARICAMENTO DATASET {stress_type.upper()} ===")
    print(f"Campioni totali: {X.shape[0]}")
    print(f"Features per campione: {X.shape[1]}")

    # Reshape: (n_samples, 400) -> (n_samples, 200, 2)
    X = X.reshape(X.shape[0], 200, 2)
    print(f"Formato dati: {X.shape} = (campioni, frequenze, componenti)")

    return X, y, plant_ids


def print_statistics(X, y, plant_ids):
    """Stampa statistiche base"""
    print("\n=== STATISTICHE DATASET ===")
    print(f"Range valori: [{X.min():.4f}, {X.max():.4f}]")
    print(f"Media: {X.mean():.4f}, Std: {X.std():.4f}")

    # Statistiche separate per Real e Imaginary
    print("\nStatistiche Real Part:")
    print(f"  Range: [{X[:, :, 0].min():.4f}, {X[:, :, 0].max():.4f}]")
    print(f"  Media: {X[:, :, 0].mean():.4f}, Std: {X[:, :, 0].std():.4f}")

    print("\nStatistiche Imaginary Part:")
    print(f"  Range: [{X[:, :, 1].min():.4f}, {X[:, :, 1].max():.4f}]")
    print(f"  Media: {X[:, :, 1].mean():.4f}, Std: {X[:, :, 1].std():.4f}")

    print("\nDistribuzione classi:")
    for cls in np.unique(y):
        count = np.sum(y == cls)
        print(
            f"  {CLASS_NAMES[cls]:15} = {count:4} campioni ({count / len(y) * 100:5.1f}%)"
        )

    print("\nCampioni per pianta:")
    for plant in np.unique(plant_ids):
        count = np.sum(plant_ids == plant)
        print(f"  Pianta {plant} = {count:4} campioni")
        for cls in np.unique(y):
            count_class = np.sum((plant_ids == plant) & (y == cls))
            if count_class > 0:
                print(f"    - {CLASS_NAMES[cls]:15} = {count_class:4} campioni")


# ======================== NORMALIZZAZIONE Z-SCORE ========================


def normalize_zscore(X):
    """
    Normalizza i dati usando Z-score (StandardScaler)
    Z-score: (x - μ) / σ
    dove μ è la media e σ è la deviazione standard

    Args:
        X: array di shape (n_samples, n_frequencies, 2)

    Returns:
        X_normalized: dati normalizzati con stessa shape
        scaler: oggetto StandardScaler per eventuale inverse_transform
    """
    n_samples = X.shape[0]

    # Flatten per normalizzazione
    X_flat = X.reshape(n_samples, -1)

    # Applica Z-score normalization
    scaler = StandardScaler()
    X_normalized_flat = scaler.fit_transform(X_flat)

    # Reshape alla forma originale
    X_normalized = X_normalized_flat.reshape(X.shape)

    return X_normalized, scaler


def plot_normalization_comparison(X_original, X_normalized, y):
    """
    Confronta visivamente dati normalizzati vs non normalizzati
    """
    print("\n=== GRUPPO 5: CONFRONTO NORMALIZZAZIONE ===")
    print("Generazione confronto dati originali vs normalizzati...")
    print(f"Risoluzione: {DPI_NORMALIZATION} DPI")

    save_dir = Path("./plots/normalization_comparison")
    save_dir.mkdir(parents=True, exist_ok=True)

    frequencies = np.logspace(np.log10(100), np.log10(1e7), 200)

    # --- CONFRONTO 1: DISTRIBUZIONI ---
    fig = plt.figure(figsize=FIGSIZE_NORMALIZATION)

    # Crea 6 subplot: 3 colonne (classi) x 2 righe (originale vs normalizzato)
    for i, cls in enumerate([0, 1, 2]):
        mask = y == cls

        # Subplot originali (riga superiore)
        ax1 = plt.subplot(2, 3, i + 1)

        # Istogramma dati originali
        real_orig = X_original[mask, :, 0].flatten()
        imag_orig = X_original[mask, :, 1].flatten()

        ax1.hist(
            real_orig, bins=50, alpha=0.5, color="blue", label="Real", density=True
        )
        ax1.hist(imag_orig, bins=50, alpha=0.5, color="red", label="Imag", density=True)
        ax1.set_title(
            f"{CLASS_NAMES[cls]} - Dati Originali", fontsize=12, fontweight="bold"
        )
        ax1.set_xlabel("Valore Impedenza", fontsize=10)
        ax1.set_ylabel("Densità", fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Aggiungi statistiche
        ax1.text(
            0.02,
            0.98,
            f"Real: μ={real_orig.mean():.2f}, σ={real_orig.std():.2f}\n"
            + f"Imag: μ={imag_orig.mean():.2f}, σ={imag_orig.std():.2f}",
            transform=ax1.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Subplot normalizzati (riga inferiore)
        ax2 = plt.subplot(2, 3, i + 4)

        # Istogramma dati normalizzati
        real_norm = X_normalized[mask, :, 0].flatten()
        imag_norm = X_normalized[mask, :, 1].flatten()

        ax2.hist(
            real_norm, bins=50, alpha=0.5, color="blue", label="Real", density=True
        )
        ax2.hist(imag_norm, bins=50, alpha=0.5, color="red", label="Imag", density=True)
        ax2.set_title(
            f"{CLASS_NAMES[cls]} - Dati Normalizzati (Z-score)",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xlabel("Valore Z-score", fontsize=10)
        ax2.set_ylabel("Densità", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Aggiungi statistiche
        ax2.text(
            0.02,
            0.98,
            f"Real: μ={real_norm.mean():.2f}, σ={real_norm.std():.2f}\n"
            + f"Imag: μ={imag_norm.mean():.2f}, σ={imag_norm.std():.2f}",
            transform=ax2.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
        )

    plt.suptitle(
        f"Confronto Distribuzioni: Originale vs Normalizzato - {STRESS_TYPE.upper()} Stress",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    filename = save_dir / f"distribution_comparison_{STRESS_TYPE}.png"
    plt.savefig(filename, dpi=DPI_NORMALIZATION, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename.name}")

    # --- CONFRONTO 2: SEPARABILITÀ CLASSI ---
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_NORMALIZATION)

    # Plot 1: Scatter originale (frequenze ottimali)
    X_opt_orig = X_original[:, OPTIMAL_FREQ_IDX, :]
    for cls in [0, 1, 2]:
        mask = y == cls
        real = X_opt_orig[mask, :, 0].flatten()
        imag = X_opt_orig[mask, :, 1].flatten()
        axes[0, 0].scatter(
            real,
            imag,
            c=CLASS_COLORS[cls],
            label=CLASS_NAMES[cls],
            alpha=0.4,
            s=20,
            edgecolors="none",
        )

    axes[0, 0].set_xlabel("Real Part (originale)", fontsize=11)
    axes[0, 0].set_ylabel("Imaginary Part (originale)", fontsize=11)
    axes[0, 0].set_title(
        f"Dati Originali - Frequenze {FREQ_RANGE}", fontsize=12, fontweight="bold"
    )
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Scatter normalizzato (frequenze ottimali)
    X_opt_norm = X_normalized[:, OPTIMAL_FREQ_IDX, :]
    for cls in [0, 1, 2]:
        mask = y == cls
        real = X_opt_norm[mask, :, 0].flatten()
        imag = X_opt_norm[mask, :, 1].flatten()
        axes[0, 1].scatter(
            real,
            imag,
            c=CLASS_COLORS[cls],
            label=CLASS_NAMES[cls],
            alpha=0.4,
            s=20,
            edgecolors="none",
        )

    axes[0, 1].set_xlabel("Real Part (Z-score)", fontsize=11)
    axes[0, 1].set_ylabel("Imaginary Part (Z-score)", fontsize=11)
    axes[0, 1].set_title(
        f"Dati Normalizzati - Frequenze {FREQ_RANGE}", fontsize=12, fontweight="bold"
    )
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Trend originale
    for cls in [0, 1, 2]:
        mask = y == cls
        mean_spectrum_real = X_original[mask, :, 0].mean(axis=0)
        axes[1, 0].semilogx(
            frequencies,
            mean_spectrum_real,
            color=CLASS_COLORS[cls],
            linewidth=2,
            label=CLASS_NAMES[cls],
            alpha=0.8,
        )

    axes[1, 0].set_xlabel("Frequenza (Hz)", fontsize=11)
    axes[1, 0].set_ylabel("Real Part Media (originale)", fontsize=11)
    axes[1, 0].set_title(
        "Spettro Medio - Dati Originali", fontsize=12, fontweight="bold"
    )
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Evidenzia frequenze ottimali
    for freq_idx in OPTIMAL_FREQ_IDX:
        axes[1, 0].axvline(
            frequencies[freq_idx],
            color="green",
            linestyle=":",
            alpha=0.5,
            linewidth=1.5,
        )

    # Plot 4: Trend normalizzato
    for cls in [0, 1, 2]:
        mask = y == cls
        mean_spectrum_real = X_normalized[mask, :, 0].mean(axis=0)
        axes[1, 1].semilogx(
            frequencies,
            mean_spectrum_real,
            color=CLASS_COLORS[cls],
            linewidth=2,
            label=CLASS_NAMES[cls],
            alpha=0.8,
        )

    axes[1, 1].set_xlabel("Frequenza (Hz)", fontsize=11)
    axes[1, 1].set_ylabel("Real Part Media (Z-score)", fontsize=11)
    axes[1, 1].set_title(
        "Spettro Medio - Dati Normalizzati", fontsize=12, fontweight="bold"
    )
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    # Evidenzia frequenze ottimali
    for freq_idx in OPTIMAL_FREQ_IDX:
        axes[1, 1].axvline(
            frequencies[freq_idx],
            color="green",
            linestyle=":",
            alpha=0.5,
            linewidth=1.5,
        )

    plt.suptitle(
        "Confronto Separabilità Classi: Originale vs Normalizzato",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    filename = save_dir / f"class_separation_comparison_{STRESS_TYPE}.png"
    plt.savefig(filename, dpi=DPI_NORMALIZATION, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename.name}")

    print("  Completato!")


def print_normalization_analysis(X_original, X_normalized):
    """
    Stampa analisi dettagliata della normalizzazione
    """
    print("\n" + "=" * 70)
    print(" ANALISI NORMALIZZAZIONE Z-SCORE")
    print("=" * 70)

    print("\n📈 EFFETTO SUI TUOI DATI:")
    print("=" * 50)

    # Calcola statistiche
    orig_real_range = X_original[:, :, 0].max() - X_original[:, :, 0].min()
    orig_imag_range = X_original[:, :, 1].max() - X_original[:, :, 1].min()
    norm_real_range = X_normalized[:, :, 0].max() - X_normalized[:, :, 0].min()
    norm_imag_range = X_normalized[:, :, 1].max() - X_normalized[:, :, 1].min()

    print(f"Range Real Part:     {orig_real_range:.2f} → {norm_real_range:.2f}")
    print(f"Range Imaginary Part: {orig_imag_range:.2f} → {norm_imag_range:.2f}")
    print("Rapporto range (Real/Imag):")
    print(f"  Originale:    {orig_real_range / orig_imag_range:.2f}")
    print(f"  Normalizzato: {norm_real_range / norm_imag_range:.2f}")


# ======================== GRUPPO 1: GROUPED_BY_PLANT_CLASS ========================


def plot_grouped_by_plant_class(X, y, plant_ids):
    """
    GRUPPO 1: Genera un file per ogni combinazione pianta/classe
    Ogni file contiene 8 subplot con tutti i campioni di quella pianta/classe
    """
    print("\n=== GRUPPO 1: GROUPED_BY_PLANT_CLASS ===")
    print("Generazione grafici per pianta e classe (8 subplot per file)...")
    print(
        f"Risoluzione: {DPI_GROUPED} DPI | Dimensioni: {FIGSIZE_GROUPED[0]}x{FIGSIZE_GROUPED[1]}"
    )

    save_dir = Path("./plots/grouped_by_plant_class")
    save_dir.mkdir(parents=True, exist_ok=True)

    frequencies = np.logspace(np.log10(100), np.log10(1e7), 200)

    for plant in np.unique(plant_ids):
        for cls in np.unique(y):
            mask = (plant_ids == plant) & (y == cls)
            indices = np.where(mask)[0]

            if len(indices) == 0:
                continue

            # Nome file chiaro: plant_P0_class_0_Control.png
            class_name = CLASS_NAMES[cls]
            filename = save_dir / f"plant_{plant}_class_{cls}_{class_name}.png"

            # Crea figura con 8 subplot (4x2)
            fig, axes = plt.subplots(4, 2, figsize=FIGSIZE_GROUPED)
            axes = axes.flatten()

            # Dividi campioni in 8 gruppi
            samples_per_subplot = max(1, len(indices) // 8)

            for subplot_idx in range(8):
                ax = axes[subplot_idx]

                # Calcola indici per questo subplot
                start = subplot_idx * samples_per_subplot
                if subplot_idx == 7:  # Ultimo subplot prende il resto
                    end = len(indices)
                else:
                    end = (subplot_idx + 1) * samples_per_subplot

                group_indices = indices[start:end]

                if len(group_indices) == 0:
                    ax.axis("off")
                    continue

                # Plot tutti i campioni del gruppo (semi-trasparenti)
                for idx in group_indices:
                    real = X[idx, :, 0]
                    imag = X[idx, :, 1]
                    # Real in linea continua
                    ax.semilogx(
                        frequencies,
                        real,
                        color=CLASS_COLORS[cls],
                        alpha=0.15,
                        linewidth=0.8,
                        linestyle="-",
                    )
                    # Imaginary in linea tratteggiata
                    ax.semilogx(
                        frequencies,
                        imag,
                        color="purple",
                        alpha=0.15,
                        linewidth=0.8,
                        linestyle="--",
                    )

                # Media del gruppo (linea SPESSA)
                real_mean = X[group_indices, :, 0].mean(axis=0)
                imag_mean = X[group_indices, :, 1].mean(axis=0)
                ax.semilogx(
                    frequencies,
                    real_mean,
                    "b-",
                    linewidth=2.5,
                    label="Media Real",
                    alpha=0.9,
                )
                ax.semilogx(
                    frequencies,
                    imag_mean,
                    "m--",
                    linewidth=2.5,
                    label="Media Imag",
                    alpha=0.9,
                )

                # Evidenzia frequenze ottimali (linee verdi punteggiate)
                for freq_idx in OPTIMAL_FREQ_IDX:
                    ax.axvline(
                        frequencies[freq_idx],
                        color="green",
                        linestyle=":",
                        alpha=0.5,
                        linewidth=1.5,
                    )

                ax.set_xlabel("Frequenza (Hz)", fontsize=10)
                ax.set_ylabel("Impedenza", fontsize=10)
                ax.set_title(
                    f"Gruppo {subplot_idx + 1} ({len(group_indices)} campioni)",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.legend(loc="best", fontsize=8)
                ax.grid(True, alpha=0.3)

            # Titolo generale
            fig.suptitle(
                f"Pianta {plant} | {class_name} | Totale: {len(indices)} campioni",
                fontsize=15,
                fontweight="bold",
            )
            plt.tight_layout()

            plt.savefig(filename, dpi=DPI_GROUPED, bbox_inches="tight")
            plt.close()

            print(f"  ✓ {filename.name} ({len(indices)} campioni)")

    print("  Completato!")


# ======================== GRUPPO 2: OPTIMAL_FREQUENCIES ========================


def plot_optimal_frequencies(X, y):
    """
    GRUPPO 2: Due file per analisi frequenze ottimali
    1. comparison: scatter plot e box plot
    2. trends: andamento attraverso le 3 frequenze
    """
    print("\n=== GRUPPO 2: OPTIMAL_FREQUENCIES ===")
    print("Generazione analisi frequenze ottimali...")
    print(f"Risoluzione: {DPI_OPTIMAL} DPI")

    save_dir = Path("./plots/optimal_frequencies")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Estrai solo le frequenze ottimali
    X_opt = X[:, OPTIMAL_FREQ_IDX, :]

    # --- FILE 1: COMPARISON ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_OPTIMAL_COMP)

    # Scatter plot Real vs Imaginary
    for cls in np.unique(y):
        mask = y == cls
        real = X_opt[mask, :, 0].flatten()
        imag = X_opt[mask, :, 1].flatten()
        ax1.scatter(
            real,
            imag,
            c=CLASS_COLORS[cls],
            label=CLASS_NAMES[cls],
            alpha=0.4,
            s=20,
            edgecolors="none",
        )

    ax1.set_xlabel("Real Part", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Imaginary Part", fontsize=13, fontweight="bold")
    ax1.set_title(f"Separazione classi alle frequenze {FREQ_RANGE}", fontsize=14)
    ax1.legend(markerscale=2, fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)

    # Box plot per classe
    data_real = []
    data_imag = []
    labels = []

    for cls in np.unique(y):
        mask = y == cls
        data_real.append(X_opt[mask, :, 0].flatten())
        data_imag.append(X_opt[mask, :, 1].flatten())
        labels.append(CLASS_NAMES[cls])

    positions = np.arange(len(labels)) * 2
    bp1 = ax2.boxplot(
        data_real,
        positions=positions - 0.4,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", linewidth=2),
        medianprops=dict(color="darkblue", linewidth=3),
    )
    bp2 = ax2.boxplot(
        data_imag,
        positions=positions + 0.4,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightcoral", linewidth=2),
        medianprops=dict(color="darkred", linewidth=3),
    )

    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.set_ylabel("Impedenza", fontsize=13, fontweight="bold")
    ax2.set_title("Distribuzione Real vs Imaginary per Classe", fontsize=14)
    ax2.legend(
        [bp1["boxes"][0], bp2["boxes"][0]], ["Real Part", "Imaginary Part"], fontsize=12
    )
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.tick_params(labelsize=11)

    plt.suptitle(
        f"Analisi Frequenze Ottimali - {STRESS_TYPE.upper()} Stress",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    filename = save_dir / f"optimal_frequencies_comparison_{STRESS_TYPE}.png"
    plt.savefig(filename, dpi=DPI_OPTIMAL, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename.name}")

    # --- FILE 2: TRENDS ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_OPTIMAL_TRENDS)

    # Frequenze ottimali effettive
    frequencies_all = np.logspace(np.log10(100), np.log10(1e7), 200)
    frequencies_opt = frequencies_all[OPTIMAL_FREQ_IDX]

    for cls in np.unique(y):
        mask = y == cls

        # Real Part
        real_mean = X_opt[mask, :, 0].mean(axis=0)
        real_std = X_opt[mask, :, 0].std(axis=0)
        ax1.plot(
            frequencies_opt,
            real_mean,
            color=CLASS_COLORS[cls],
            linewidth=3,
            marker="o",
            markersize=12,
            label=CLASS_NAMES[cls],
            alpha=0.8,
        )
        ax1.fill_between(
            frequencies_opt,
            real_mean - real_std,
            real_mean + real_std,
            color=CLASS_COLORS[cls],
            alpha=0.15,
        )

        # Imaginary Part
        imag_mean = X_opt[mask, :, 1].mean(axis=0)
        imag_std = X_opt[mask, :, 1].std(axis=0)
        ax2.plot(
            frequencies_opt,
            imag_mean,
            color=CLASS_COLORS[cls],
            linewidth=3,
            marker="s",
            markersize=12,
            label=CLASS_NAMES[cls],
            alpha=0.8,
        )
        ax2.fill_between(
            frequencies_opt,
            imag_mean - imag_std,
            imag_mean + imag_std,
            color=CLASS_COLORS[cls],
            alpha=0.15,
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("Frequenza (Hz)", fontsize=12)
    ax1.set_ylabel("Real Part (Media ± Std)", fontsize=12)
    ax1.set_title("Trend Real Part attraverso le 3 frequenze ottimali", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)

    ax2.set_xscale("log")
    ax2.set_xlabel("Frequenza (Hz)", fontsize=12)
    ax2.set_ylabel("Imaginary Part (Media ± Std)", fontsize=12)
    ax2.set_title(
        "Trend Imaginary Part attraverso le 3 frequenze ottimali", fontsize=13
    )
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)

    plt.suptitle(
        f"Trend Frequenze Ottimali ({FREQ_RANGE}) - {STRESS_TYPE.upper()} Stress",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    filename = save_dir / f"optimal_frequencies_trends_{STRESS_TYPE}.png"
    plt.savefig(filename, dpi=DPI_OPTIMAL, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename.name}")
    print("  Completato!")


# ======================== GRUPPO 3: FULL_SPECTRUM ========================


def plot_full_spectrum(X, y):
    """
    GRUPPO 3: Due file per analisi spettro completo (tutte 200 frequenze)
    1. comparison: come gruppo 2 ma con tutti i dati
    2. trends: curve continue attraverso tutto lo spettro
    """
    print("\n=== GRUPPO 3: FULL_SPECTRUM ===")
    print("Generazione analisi spettro completo...")
    print(f"Risoluzione: {DPI_FULL_SPECTRUM} DPI")

    save_dir = Path("./plots/full_spectrum")
    save_dir.mkdir(parents=True, exist_ok=True)

    frequencies = np.logspace(np.log10(100), np.log10(1e7), 200)

    # --- FILE 1: COMPARISON ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_FULL_SPEC_COMP)

    # Scatter plot (tutti i 400 valori)
    for cls in np.unique(y):
        mask = y == cls
        real = X[mask, :, 0].flatten()
        imag = X[mask, :, 1].flatten()
        ax1.scatter(
            real,
            imag,
            c=CLASS_COLORS[cls],
            label=CLASS_NAMES[cls],
            alpha=0.3,
            s=5,
            edgecolors="none",
        )

    ax1.set_xlabel("Real Part", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Imaginary Part", fontsize=13, fontweight="bold")
    ax1.set_title("Full Spectrum: Real vs Imaginary (100 Hz - 10 MHz)", fontsize=14)
    ax1.legend(markerscale=4, fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)

    # Box plot
    data_real = []
    data_imag = []
    labels = []

    for cls in np.unique(y):
        mask = y == cls
        data_real.append(X[mask, :, 0].flatten())
        data_imag.append(X[mask, :, 1].flatten())
        labels.append(CLASS_NAMES[cls])

    positions = np.arange(len(labels)) * 2
    bp1 = ax2.boxplot(
        data_real,
        positions=positions - 0.4,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", linewidth=2),
        medianprops=dict(color="darkblue", linewidth=3),
    )
    bp2 = ax2.boxplot(
        data_imag,
        positions=positions + 0.4,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightcoral", linewidth=2),
        medianprops=dict(color="darkred", linewidth=3),
    )

    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.set_ylabel("Impedenza", fontsize=13, fontweight="bold")
    ax2.set_title("Full Spectrum: Distribuzione per Classe", fontsize=14)
    ax2.legend(
        [bp1["boxes"][0], bp2["boxes"][0]], ["Real Part", "Imaginary Part"], fontsize=12
    )
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.tick_params(labelsize=11)

    plt.suptitle(
        f"Full Spectrum Analysis - {STRESS_TYPE.upper()} Stress",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    filename = save_dir / f"full_spectrum_comparison_{STRESS_TYPE}.png"
    plt.savefig(filename, dpi=DPI_FULL_SPECTRUM, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename.name}")

    # --- FILE 2: TRENDS ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_FULL_SPEC_TRENDS)

    for cls in np.unique(y):
        mask = y == cls
        color = CLASS_COLORS[cls]
        label = CLASS_NAMES[cls]

        # Real Part
        real_mean = X[mask, :, 0].mean(axis=0)
        real_std = X[mask, :, 0].std(axis=0)
        ax1.semilogx(
            frequencies, real_mean, color=color, linewidth=2.5, label=label, alpha=0.8
        )
        ax1.fill_between(
            frequencies,
            real_mean - real_std,
            real_mean + real_std,
            color=color,
            alpha=0.1,
        )

        # Imaginary Part
        imag_mean = X[mask, :, 1].mean(axis=0)
        imag_std = X[mask, :, 1].std(axis=0)
        ax2.semilogx(
            frequencies, imag_mean, color=color, linewidth=2.5, label=label, alpha=0.8
        )
        ax2.fill_between(
            frequencies,
            imag_mean - imag_std,
            imag_mean + imag_std,
            color=color,
            alpha=0.1,
        )

    # Evidenzia frequenze ottimali
    for freq_idx in OPTIMAL_FREQ_IDX:
        ax1.axvline(
            frequencies[freq_idx], color="green", linestyle=":", alpha=0.5, linewidth=2
        )
        ax2.axvline(
            frequencies[freq_idx], color="green", linestyle=":", alpha=0.5, linewidth=2
        )

    ax1.set_xlabel("Frequenza (Hz)", fontsize=12)
    ax1.set_ylabel("Real Part (Media ± Std)", fontsize=12)
    ax1.set_title("Full Spectrum - Real Part per Classe", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)

    ax2.set_xlabel("Frequenza (Hz)", fontsize=12)
    ax2.set_ylabel("Imaginary Part (Media ± Std)", fontsize=12)
    ax2.set_title("Full Spectrum - Imaginary Part per Classe", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)

    plt.suptitle(
        f"Trend di Frequenza (Full Spectrum) - {STRESS_TYPE.upper()} Stress",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    filename = save_dir / f"full_spectrum_trends_{STRESS_TYPE}.png"
    plt.savefig(filename, dpi=DPI_FULL_SPECTRUM, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename.name}")
    print("  Completato!")


# ======================== GRUPPO 4: INDIVIDUAL_SAMPLES ========================


def plot_individual_samples(X, y, plant_ids):
    """
    GRUPPO 4: Un file per ogni singolo campione (2016 file totali)
    """
    print("\n=== GRUPPO 4: INDIVIDUAL_SAMPLES ===")
    print(f"Generazione {len(X)} grafici individuali...")
    print(
        f"Risoluzione: {DPI_INDIVIDUAL} DPI | Dimensioni: {FIGSIZE_INDIVIDUAL[0]}x{FIGSIZE_INDIVIDUAL[1]}"
    )
    print("ATTENZIONE: Questa operazione richiederà 5-10 minuti...")

    save_dir = Path("./plots/individual_samples")
    save_dir.mkdir(parents=True, exist_ok=True)

    frequencies = np.logspace(np.log10(100), np.log10(1e7), 200)

    for idx in range(len(X)):
        real = X[idx, :, 0]
        imag = X[idx, :, 1]
        magnitude = np.sqrt(real**2 + imag**2)

        plant_id = plant_ids[idx]
        label = int(y[idx])
        class_name = CLASS_NAMES[label]
        color = CLASS_COLORS[label]

        # Nome file chiaro: sample_0000_plant_P0_class_0.png
        filename = save_dir / f"sample_{idx:04d}_plant_{plant_id}_class_{label}.png"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_INDIVIDUAL)

        # Subplot 1: Real e Imaginary
        ax1.semilogx(
            frequencies, real, "b-", label="Real Part", linewidth=1.5, alpha=0.8
        )
        ax1.semilogx(
            frequencies, imag, "r-", label="Imaginary Part", linewidth=1.5, alpha=0.8
        )

        # Evidenzia frequenze ottimali
        for freq_idx in OPTIMAL_FREQ_IDX:
            ax1.axvline(
                frequencies[freq_idx],
                color="green",
                linestyle="--",
                alpha=0.5,
                linewidth=1,
            )

        ax1.set_xlabel("Frequenza (Hz)", fontsize=11)
        ax1.set_ylabel("Impedenza", fontsize=11)
        ax1.set_title(
            f"Campione #{idx} | Pianta {plant_id} | {class_name}",
            fontsize=12,
            fontweight="bold",
            color=color,
        )
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=10)

        # Subplot 2: Magnitudine
        ax2.semilogx(frequencies, magnitude, color=color, linewidth=2, alpha=0.8)

        # Evidenzia frequenze ottimali
        for freq_idx in OPTIMAL_FREQ_IDX:
            ax2.axvline(
                frequencies[freq_idx],
                color="green",
                linestyle="--",
                alpha=0.5,
                linewidth=1,
            )

        ax2.set_xlabel("Frequenza (Hz)", fontsize=11)
        ax2.set_ylabel("Magnitudine |Z|", fontsize=11)
        ax2.set_title(
            f"Magnitudine Impedenza | Freq. ottimali: {FREQ_RANGE}", fontsize=11
        )
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=10)

        plt.tight_layout()
        plt.savefig(filename, dpi=DPI_INDIVIDUAL, bbox_inches="tight")
        plt.close()

        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(X)} completati...")

    print(f"  ✓ Generati {len(X)} grafici individuali!")
    print("  Completato!")


# ======================== MAIN ========================


def main():
    print("\n" + "=" * 60)
    print(" ANALISI DATASET PIANTE - STRESS DETECTION")
    print(" CON NORMALIZZAZIONE Z-SCORE")
    print("=" * 60)

    # Input configurazione
    print("\nCONFIGURAZIONE:")
    stress_input = (
        input("Quale dataset analizzare? (water/iron) [default: water]: ")
        .strip()
        .lower()
    )
    if stress_input in ["water", "iron"]:
        global STRESS_TYPE, OPTIMAL_FREQ_IDX, FREQ_RANGE
        STRESS_TYPE = stress_input
        if STRESS_TYPE == "iron":
            OPTIMAL_FREQ_IDX = [197, 198, 199]
            FREQ_RANGE = "4.7-10 MHz"

    individual_input = (
        input("\nGenerare anche i 2016 grafici individuali? (s/n) [default: n]: ")
        .strip()
        .lower()
    )
    GENERATE_INDIVIDUAL = individual_input == "s"

    if GENERATE_INDIVIDUAL:
        print("\nATTENZIONE: La generazione di 2016 grafici richiederà:")
        print("   - Tempo: 1-5 minuti")
        print("   - Spazio: ~500-700 MB")
        confirm = input("   Continuare? (s/n): ").strip().lower()
        if confirm != "s":
            GENERATE_INDIVIDUAL = False

    # Carica dataset
    try:
        X, y, plant_ids = load_dataset(STRESS_TYPE)
    except FileNotFoundError:
        print(
            f"\nERRORE: File ./data/{STRESS_TYPE.capitalize()}_Stress.npz non trovato!"
        )
        print("   Verificare che il file sia nella cartella ./data/")
        return

    # Stampa statistiche originali
    print_statistics(X, y, plant_ids)

    # NORMALIZZAZIONE Z-SCORE
    print("\n" + "=" * 60)
    print(" NORMALIZZAZIONE Z-SCORE")
    print("=" * 60)

    X_normalized, scaler = normalize_zscore(X)

    print("\n✓ Normalizzazione completata!")
    print("  Formula applicata: Z = (X - μ) / σ")
    print("  Nuova media ≈ 0, Nuova std ≈ 1")

    # Stampa statistiche normalizzate
    print("\n=== STATISTICHE DATI NORMALIZZATI ===")
    print(f"Range valori: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")
    print(f"Media: {X_normalized.mean():.6f} (≈0)")
    print(f"Std: {X_normalized.std():.6f} (≈1)")

    # Genera confronto normalizzazione
    if GENERATE_NORMALIZATION_COMPARISON:
        plot_normalization_comparison(X, X_normalized, y)
        print_normalization_analysis(X, X_normalized)

    # Chiedi all'utente quale dataset usare per i grafici standard
    print("\n" + "=" * 60)
    use_normalized = (
        input("\nUsare dati normalizzati per i grafici standard? (s/n) [default: n]: ")
        .strip()
        .lower()
    )

    if use_normalized == "s":
        X_to_use = X_normalized
        print("→ Utilizzo dati NORMALIZZATI per i grafici")
    else:
        X_to_use = X
        print("→ Utilizzo dati ORIGINALI per i grafici")

    # Genera grafici standard (con dati scelti dall'utente)
    print("\n" + "=" * 60)
    print(" GENERAZIONE GRAFICI STANDARD")
    print("=" * 60)

    # GRUPPO 1: Grafici per pianta/classe
    plot_grouped_by_plant_class(X_to_use, y, plant_ids)

    # GRUPPO 2: Frequenze ottimali
    plot_optimal_frequencies(X_to_use, y)

    # GRUPPO 3: Full spectrum
    plot_full_spectrum(X_to_use, y)

    # GRUPPO 4: Grafici individuali (opzionale)
    if GENERATE_INDIVIDUAL:
        plot_individual_samples(X, y, plant_ids)

    # Riepilogo finale
    print("\n" + "=" * 60)
    print(" ANALISI COMPLETATA!")
    print("=" * 60)
    print("\nGRAFICI GENERATI:")
    print(" ./plots/normalization_comparison/ (2 file di confronto)")
    print(" ./plots/grouped_by_plant_class/ (grafici raggruppati)")
    print(" ./plots/optimal_frequencies/ (analisi frequenze ottimali)")
    print(" ./plots/full_spectrum/ (analisi spettro completo)")
    if GENERATE_INDIVIDUAL:
        print(" ./plots/individual_samples/ (2016 grafici individuali)")


if __name__ == "__main__":
    main()
