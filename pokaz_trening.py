import os
import matplotlib.pyplot as plt
from GA import config as cfg
import csv
import numpy as np

# --- Konfiguracja ---
# Skrypt teraz automatycznie znajduje plik CSV
NAZWA_PLIKU_LOGU = f"log_wynikow_{cfg.MODEL_SAVE_DIR}.csv"


# --------------------

def load_data_from_csv(sciezka_pliku):
    """Przetwarza plik CSV z wynikami."""
    generations = []
    best_scores = []
    avg_scores = []

    if not os.path.exists(sciezka_pliku):
        print(f"Błąd: Plik '{sciezka_pliku}' nie istnieje.")
        print("Najpierw uruchom `mikroswiat_smart.py` (wersja v32+), aby wygenerować plik logu.")
        return None

    with open(sciezka_pliku, mode='r') as f:
        reader = csv.reader(f)
        try:
            next(reader)  # Pomiń nagłówek
        except StopIteration:
            print("Błąd: Plik logu jest pusty.")
            return None

        for row in reader:
            try:
                generations.append(int(row[0]))
                best_scores.append(float(row[1]))
                avg_scores.append(float(row[2]))
            except (ValueError, IndexError):
                print(f"Pominięto błędny wiersz: {row}")

    if not generations:
        print("Błąd: Nie znaleziono poprawnych danych w pliku logu.")
        return None

    return generations, best_scores, avg_scores


def plot_results(data):
    """Rysuje wykres na podstawie zebranych danych."""
    generations, best_scores, avg_scores = data

    plt.figure(figsize=(12, 7))

    # 1. Rysuj "Najlepszego agenta" jako półprzezroczyste tło (szum)
    plt.plot(generations, best_scores, label="Najlepszy agent (szum)", marker='o', linestyle='-', color='blue',
             alpha=0.2)

    # 2. Rysuj "Średni wynik populacji" jako główną linię
    plt.plot(generations, avg_scores, label="Średni wynik populacji (Główny trend)", linestyle='-', color='red',
             linewidth=2.5)

    # 3. Wygładź średnią linię, aby pokazać trend długoterminowy
    if len(avg_scores) >= 10:
        moving_avg = np.convolve(avg_scores, np.ones(10) / 10, mode='valid')
        # Dostosuj oś 'x' dla średniej ruchomej
        start_index = len(generations) - len(moving_avg)
        plt.plot(generations[start_index:], moving_avg, label="Trend średniej (wygładzony)", linestyle='--',
                 color='black', linewidth=2)

    plt.title(f"Postęp Ewolucji (Mózg: {cfg.MODEL_SAVE_DIR})")
    plt.xlabel("Numer Generacji")
    plt.ylabel("Wynik Fitness (Punkty)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Zapisz wykres do pliku
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