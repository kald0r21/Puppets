import os
import re
import matplotlib.pyplot as plt
import sys
import subprocess

# --- ZMIANA: Import konfiguracji ---
try:
    import config_dqn as cfg
except ImportError:
    print("Błąd: Nie znaleziono pliku 'config_dqn.py'. Upewnij się, że plik istnieje.")
    exit()

import csv
import numpy as np

# --- Konfiguracja ---
NAZWA_PLIKU_LOGU = f"log_wynikow_{cfg.MODEL_SAVE_DIR}.csv"
ILOSC_PIKOW_DO_POKAZANIA = 5  # Ile najwyższych wyników oznaczyć na wykresie


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
        print("Najpierw uruchom 'mikroswiat_dqn.py', aby wygenerować plik logu.")
        return None

    with open(sciezka_pliku, mode='r') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)  # Pomiń nagłówek
        except StopIteration:
            print("Błąd: Plik logu jest pusty.")
            return None

        num_cols = len(header)

        for row in reader:
            try:
                generations.append(int(row[0]))
                best_scores.append(float(row[1]))
                avg_scores.append(float(row[2]))

                if num_cols >= 5:
                    eaten_list.append(int(row[3]))
                    killed_list.append(int(row[4]))

            except (ValueError, IndexError):
                pass  # Ignoruj puste linie

    if not generations:
        print("Błąd: Nie znaleziono poprawnych danych w pliku logu.")
        return None

    if eaten_list:
        return generations, best_scores, avg_scores, eaten_list, killed_list
    else:
        return generations, best_scores, avg_scores


def plot_results(data):
    """Rysuje wykres na podstawie zebranych danych."""

    has_stats = len(data) == 5

    if has_stats:
        generations, best_scores, avg_scores, eaten_list, killed_list = data
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    else:
        generations, best_scores, avg_scores = data
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))

    # --- Wykres 1: Wynik (Nagroda) ---
    ax1.set_title(f"Postęp Ewolucji (Mózg: {cfg.MODEL_SAVE_DIR})")
    ax1.plot(generations, best_scores, label="Wynik Agenta (Epizod)", marker='o', linestyle='-', color='blue',
             alpha=0.3, markersize=3)
    ax1.plot(generations, avg_scores, label="Średnia (10 ost.)", linestyle='-', color='red',
             linewidth=2)

    # Oznaczanie pików
    data_points = list(zip(generations, best_scores))
    data_points.sort(key=lambda x: x[1], reverse=True)
    top_peaks = data_points[:ILOSC_PIKOW_DO_POKAZANIA]

    print("\n--- Najwyższe wyniki w historii (Piki) ---")
    for (gen, score) in top_peaks:
        print(f"  Epizod {gen}: {int(score)} pkt")
        ax1.annotate(f"{int(score)}",
                     xy=(gen, score),
                     xytext=(gen, score + (max(best_scores) * 0.1)),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                     ha='center', fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec="k", lw=1, alpha=0.7))

    ax1.set_ylabel("Wynik (Reward)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Wykres 2: Statystyki ---
    if has_stats:
        ax2.set_title("Statystyki Przeżycia")
        window_size = 20
        if len(eaten_list) >= window_size:
            eaten_smooth = np.convolve(eaten_list, np.ones(window_size) / window_size, mode='valid')
            killed_smooth = np.convolve(killed_list, np.ones(window_size) / window_size, mode='valid')
            start_index = len(generations) - len(eaten_smooth)

            ax2.plot(generations[start_index:], eaten_smooth, label=f"Zjedzone (śr. {window_size})",
                     color='green', linewidth=2)
            ax2.plot(generations[start_index:], killed_smooth, label=f"Zabite (śr. {window_size})",
                     color='purple', linewidth=2)
        else:
            ax2.plot(generations, eaten_list, label="Zjedzone", color='green', alpha=0.5)
            ax2.plot(generations, killed_list, label="Zabite", color='purple', alpha=0.5)

        ax2.set_xlabel("Numer Epizodu")
        ax2.set_ylabel("Liczba")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax1.set_xlabel("Numer Epizodu")

    plt.tight_layout()
    save_path = f"postep_treningu_{cfg.MODEL_SAVE_DIR}.png"
    plt.savefig(save_path)
    print(f"\nWykres zapisany jako: '{save_path}'")
    print("Zamknij okno wykresu, aby kontynuować...")
    plt.show()


def znajdz_i_uruchom_najlepszy(data):
    """Szuka najlepszego ZAPISANEGO modelu i proponuje jego uruchomienie."""
    if len(data) == 5:
        generations, best_scores, _, _, _ = data
    else:
        generations, best_scores, _ = data

    # Sortuj wyniki od najlepszego
    # Tworzymy listę krotek (wynik, epizod)
    ranked_episodes = sorted(zip(best_scores, generations), key=lambda x: x[0], reverse=True)

    print("\n--- Poszukiwanie najlepszego dostępnego modelu ---")

    best_available_ep = None
    best_available_score = None

    # Przeszukujemy ranking od góry, sprawdzając czy plik istnieje
    for score, ep in ranked_episodes:
        model_path = os.path.join(cfg.MODEL_SAVE_DIR, f"agent_ep_{ep}.pth")
        if os.path.exists(model_path):
            best_available_ep = ep
            best_available_score = score
            break

    if best_available_ep:
        print(f"Najlepszy ZAPISANY model znaleziono w epizodzie: {best_available_ep}")
        print(f"Wynik tego epizodu: {int(best_available_score)}")

        wybor = input(f"\nCzy chcesz uruchomić podgląd tego epizodu ({best_available_ep})? [t/n]: ").lower()
        if wybor == 't' or wybor == 'y':
            print(f"Uruchamiam obserwuj_walke.py... (Przygotuj się, by wpisać {best_available_ep})")

            # Próba uruchomienia skryptu obserwacji
            skrypt_obserwacji = "obserwuj_walke.py"
            if os.path.exists(skrypt_obserwacji):
                # Używamy sys.executable, by użyć tego samego pythona
                try:
                    subprocess.run([sys.executable, skrypt_obserwacji])
                except Exception as e:
                    print(f"Nie udało się uruchomić skryptu: {e}")
            else:
                print(f"Nie znaleziono pliku {skrypt_obserwacji} w tym folderze.")
    else:
        print("Nie znaleziono żadnych zapisanych modeli (.pth), które pasowałyby do wyników w logu.")
        print("Upewnij się, że trening trwał wystarczająco długo, by zapisać checkpointy.")


def main():
    print(f"Analizowanie wyników z pliku: '{NAZWA_PLIKU_LOGU}'...")
    dane = load_data_from_csv(NAZWA_PLIKU_LOGU)

    if dane:
        plot_results(dane)
        # Po zamknięciu wykresu:
        znajdz_i_uruchom_najlepszy(dane)


if __name__ == "__main__":
    main()