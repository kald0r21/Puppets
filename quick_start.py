#!/usr/bin/env python3
# quick_start.py - Szybkie uruchamianie różnych trybów

import sys
import os
import subprocess

MODES = {
    "1": {
        "name": "Podstawowy trening (REINFORCE)",
        "cmd": "python -m src.scripts.micro_world_vision"
    },
    "2": {
        "name": "Trening PPO (100k steps)",
        "cmd": "python -m src.scripts.train_ppo_sb3 --timesteps 100000"
    },
    "3": {
        "name": "Trening PPO (500k steps)",
        "cmd": "python -m src.scripts.train_ppo_sb3 --timesteps 500000"
    },
    "4": {
        "name": "Wizualizacja manualna",
        "cmd": "python -m src.scripts.viz_realtime --cell 8 --steps 2"
    },
    "5": {
        "name": "Wizualizacja z AI (wymaga modelu)",
        "cmd": "python -m src.scripts.viz_realtime --model checkpoints/ppo_mw_final.zip --cell 8"
    },
    "6": {
        "name": "Test modelu (terminale)",
        "cmd": "python -m src.scripts.train_ppo_sb3 --play checkpoints/ppo_mw_final.zip"
    },
    "7": {
        "name": "Tensorboard",
        "cmd": "tensorboard --logdir=./runs"
    },
}

def main():
    print("\n" + "="*60)
    print("  MicroWorld - Quick Start")
    print("="*60 + "\n")
    
    for key, mode in sorted(MODES.items()):
        print(f"  [{key}] {mode['name']}")
    
    print(f"\n  [q] Wyjście\n")
    print("="*60)
    
    choice = input("\nWybierz tryb: ").strip()
    
    if choice.lower() == 'q':
        print("Do widzenia!")
        return
    
    if choice not in MODES:
        print(f"Nieprawidłowy wybór: {choice}")
        return
    
    mode = MODES[choice]
    print(f"\n🚀 Uruchamiam: {mode['name']}")
    print(f"📝 Komenda: {mode['cmd']}\n")
    print("="*60 + "\n")
    
    try:
        subprocess.run(mode['cmd'], shell=True, check=True)
    except KeyboardInterrupt:
        print("\n\n⚠️  Przerwano przez użytkownika")
    except subprocess.CalledProcessError as e:
        print(f"\n\n❌ Błąd: {e}")

if __name__ == "__main__":
    main()
