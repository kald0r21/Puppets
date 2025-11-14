import os
import re
import matplotlib.pyplot as plt
import config as cfg  # Importujemy config, by znać nazwę folderu

# --- Konfiguracja ---
FOLDER_DO_ANALIZY = cfg.MODEL_SAVE_DIR


# --------------------

def parse_fitness_from_filename(filename):
    """
    Wyciąga numer generacji i wynik fitness z nazwy pliku.
    Wzór: gen_XX_fitness_YYYY_...
    """
    match = re.search(r"gen_(\d+)_fitness_(\d+)", filename)
    if match:
        generation = int(match.group(1))
        fitness = int(match.group(2))
        return generation, fitness
    return None, None


def load_data(folder):
    """Przetwarza wszystkie pliki w folderze i zwraca posortowane dane."""
    data = []
    if not os.path.exists(folder):
        print(f"Błąd: Folder '{folder}' nie istnieje.")
        print("Najpierw uruchom `mikroswiat_smart.py`, aby wygenerować wyniki.")
        return None

    for filename in os.listdir(folder):
        gen, fitness = parse_fitness_from_filename(filename)
        if gen is not None:
            data.append((gen, fitness))

    # Sortuj dane według generacji
    if not data:
        print(f"Błąd: Nie znaleziono żadnych plików z wynikami w folderze '{folder}'.")
        return None

    data.sort(key=lambda x: x[0])

    # Rozdziel na osobne listy
    generations = [d[0] for d in data]
    fitness_scores = [d[1] for d in data]

    return generations, fitness_scores


def plot_results(data):
    """Rysuje wykres na podstawie zebranych danych."""
    generations, fitness_scores = data

    plt.figure(figsize=(12, 7))

    plt.plot(generations, fitness_scores, label="Najlepszy agent w pokoleniu", marker='o', linestyle='-', color='blue')

    # Obliczanie i rysowanie linii trendu (średnia ruchoma)
    if len(fitness_scores) >= 5:
        # Prosta średnia ruchoma z 5 ostatnich pokoleń
        moving_avg = np.convolve(fitness_scores, np.ones(5) / 5, mode='valid')
        # Musimy dostosować oś 'x' dla średniej ruchomej
        plt.plot(generations[4:], moving_avg, label="Trend (średnia ruchoma z 5 pok.)", linestyle='--', color='red',
                 linewidth=2)

    plt.title(f"Postęp Ewolucji (Mózg: {cfg.MODEL_SAVE_DIR})")
    plt.xlabel("Numer Generacji")
    plt.ylabel("Wynik Fitness (Punkty)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Zapisz wykres do pliku
    save_path = f"postep_treningu_{cfg.MODEL_SAVE_DIR}.png"
    plt.savefig(save_path)
    print(f"\nWykres walidacyjny został zapisany jako: '{save_path}'")
    plt.show()


def main():
    print(f"Analizowanie wyników z folderu: '{FOLDER_DO_ANALIZY}'...")
    dane = load_data(FOLDER_DO_ANALIZY)

    if dane:
        plot_results(dane)


if __name__ == "__main__":
    # Musimy zaimportować numpy tylko dla średniej ruchomej
    try:
        import numpy as np
    except ImportError:
        print("Błąd: Ten skrypt wymaga 'numpy'. Zainstaluj go: pip install numpy")
        exit()

    main()