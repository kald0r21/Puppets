import os
import re
import matplotlib.pyplot as plt

# --- ZMIANA: Skrypt, który wysłałeś, importował `config` z `GA` ---
# Upewnij się, że ten import jest poprawny dla Ciebie
# Jeśli Twój config nazywa się `config_smart.py`, zmień poniższą linię:
try:
    import config_dqn as cfg
except ImportError:
    print("Błąd: Nie znaleziono pliku 'config_smart.py'. Upewnij się, że plik istnieje.")
    exit()

import csv
import numpy as np

# --- Konfiguracja ---
# Skrypt teraz automatycznie znajduje plik CSV
NAZWA_PLIKU_LOGU = f"log_wynikow_{cfg.MODEL_SAVE_DIR}.csv"
ILOSC_PIKOW_DO_POKAZANIA = 5  # Ile najwyższych wyników oznaczyć


# --------------------

def load_data_from_csv(sciezka_pliku):
    """Przetwarza plik CSV z wynikami."""
    generations = []
    best_scores = []
    avg_scores = []
    eaten_list = []
    killed_list = []

    if not os.path.exists(sciezka_pliku):
        print(f"Błąd: Plik '{sciezka_pliku}' nie istnieje.")
        print("Najpierw uruchom `mikroswiat_smart.py` (wersja v32+), aby wygenerować plik logu.")
        return None

    with open(sciezka_pliku, mode='r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)  # Pomiń nagłówek
        except StopIteration:
            print("Błąd: Plik logu jest pusty.")
            return None

        # Sprawdź, ile kolumn ma plik
        num_cols = len(header)

        for row in reader:
            try:
                generations.append(int(row[0]))
                best_scores.append(float(row[1]))
                avg_scores.append(float(row[2]))

                # Dodano obsługę plików CSV z różną liczbą kolumn
                if num_cols >= 5:
                    eaten_list.append(int(row[3]))
                    killed_list.append(int(row[4]))

            except (ValueError, IndexError):
                print(f"Pominięto błędny wiersz: {row}")

    if not generations:
        print("Błąd: Nie znaleziono poprawnych danych w pliku logu.")
        return None

    # Zwróć dane w zależności od tego, co było w pliku
    if eaten_list:
        return generations, best_scores, avg_scores, eaten_list, killed_list
    else:
        return generations, best_scores, avg_scores


def plot_results(data):
    """Rysuje wykres na podstawie zebranych danych."""

    # Sprawdź, czy dane zawierają statystyki jedzenia/zabijania
    has_stats = len(data) == 5

    if has_stats:
        generations, best_scores, avg_scores, eaten_list, killed_list = data
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    else:
        generations, best_scores, avg_scores = data
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))

    # --- Wykres 1: Wynik (Nagroda) Agenta ---
    ax1.set_title(f"Postęp Ewolucji (Mózg: {cfg.MODEL_SAVE_DIR})")
    ax1.plot(generations, best_scores, label="Najlepszy agent (szum)", marker='o', linestyle='-', color='blue',
             alpha=0.2)
    ax1.plot(generations, avg_scores, label="Średni wynik populacji (Główny trend)", linestyle='-', color='red',
             linewidth=2.5)

    # --- NOWOŚĆ: OZNACZANIE PIKÓW ---
    data_points = list(zip(generations, best_scores))
    data_points.sort(key=lambda x: x[1], reverse=True)  # Sortuj wg wyniku

    top_peaks = data_points[:ILOSC_PIKOW_DO_POKAZANIA]

    print("\n--- Najwyższe wyniki (Piki) ---")
    for (gen, score) in top_peaks:
        print(f"  Generacja {gen}: {int(score)} pkt")
        ax1.annotate(f"Gen: {gen}\nWynik: {int(score)}",
                     xy=(gen, score),
                     xytext=(gen, score + 0.1 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])),  # Offset 10% w górę
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                     ha='center',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="k", lw=1, alpha=0.7))
    # ---------------------------------

    ax1.set_ylabel("Wynik Fitness (Punkty)")
    ax1.legend()
    ax1.grid(True)

    # --- Wykres 2: Strategia Agenta (jeśli dane istnieją) ---
    if has_stats:
        ax2.set_title("Statystyki Przeżycia (Trend)")
        window_size = 20
        if len(eaten_list) >= window_size:
            eaten_smooth = np.convolve(eaten_list, np.ones(window_size) / window_size, mode='valid')
            killed_smooth = np.convolve(killed_list, np.ones(window_size) / window_size, mode='valid')
            start_index = len(generations) - len(eaten_smooth)

            ax2.plot(generations[start_index:], eaten_smooth, label=f"Zjedzone (śr. ruchoma {window_size})",
                     color='green', linewidth=2)
            ax2.plot(generations[start_index:], killed_smooth, label=f"Zabite (śr. ruchoma {window_size})",
                     color='purple', linewidth=2)
        else:
            ax2.plot(generations, eaten_list, label="Zjedzone", color='green', alpha=0.5)
            ax2.plot(generations, killed_list, label="Zabite", color='purple', alpha=0.5)

        ax2.set_xlabel("Numer Generacji")
        ax2.set_ylabel("Liczba")
        ax2.legend()
        ax2.grid(True)
    else:
        ax1.set_xlabel("Numer Generacji")  # Dodaj etykietę X, jeśli jest tylko 1 wykres

    plt.tight_layout()
    save_path = f"postep_treningu_PELNY_{cfg.MODEL_SAVE_DIR}.png"
    plt.savefig(save_path)
    print(f"\nWykres walidacyjny został zapisany jako: '{save_path}'")
    plt.show()


def main():
    print(f"Analizowanie wyników z pliku: '{NAZWA_PLIKU_LOGU}'...")
    dane = load_data_from_csv(NAZWA_PLIKU_LOGU)

    if dane:
        plot_results(dane)


if __name__ == "__main__":
    main()