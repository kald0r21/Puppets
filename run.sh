#!/bin/bash
# run.sh - Instrukcje uruchamiania modułów MicroWorld

echo "=== MicroWorld - Instrukcje uruchamiania ==="
echo ""

echo "1. PODSTAWOWY TRENING (standalone, REINFORCE/A2C):"
echo "   python -m src.scripts.micro_world_vision"
echo "   python -m src.scripts.micro_world_vision --render"
echo ""

echo "2. TRENING PPO (Stable-Baselines3):"
echo "   python -m src.scripts.train_ppo_sb3 --timesteps 500000"
echo "   python -m src.scripts.train_ppo_sb3 --timesteps 500000 --tb ./runs/exp1 --ckpt ./checkpoints/exp1"
echo ""

echo "3. TESTOWANIE WYTRENOWANEGO MODELU:"
echo "   python -m src.scripts.train_ppo_sb3 --play checkpoints/ppo_mw_final.zip"
echo "   python -m src.scripts.train_ppo_sb3 --play checkpoints/best_model.zip"
echo ""

echo "4. WIZUALIZACJA REALTIME (manualna kontrola):"
echo "   python -m src.scripts.viz_realtime"
echo "   python -m src.scripts.viz_realtime --cell 8 --steps 4"
echo ""

echo "5. WIZUALIZACJA Z AI:"
echo "   python -m src.scripts.viz_realtime --model checkpoints/ppo_mw_final.zip"
echo "   python -m src.scripts.viz_realtime --model checkpoints/ppo_mw_final.zip --cell 10 --steps 2"
echo ""

echo "6. TENSORBOARD (monitoring treningu):"
echo "   tensorboard --logdir=./runs"
echo "   tensorboard --logdir=./runs/micro_world_ppo"
echo ""

echo "=== Skróty klawiszowe (viz_realtime.py) ==="
echo "   Strzałki / WASD: ruch (tryb manualny)"
echo "   P: pauza/wznów"
echo "   SPACJA: krok w trybie pauzy"
echo "   R: reset środowiska"
echo "   +/-: zmiana prędkości symulacji"
echo "   ESC/Q: wyjście"
echo ""

echo "=== Instalacja zależności ==="
echo "   pip install numpy torch stable-baselines3[extra] pygame tensorboard gymnasium"
echo ""