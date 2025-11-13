# src/scripts/train_ppo_sb3.py
# Wersja 5.3: Zwiększenie obciążenia GPU (więcej epok, większy batch)

import argparse, os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import configure

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.micro_world.env_gym import MicroWorldVisionEnv
from src.scripts.custom_policy import SmallCNN

# Definiowanie ścieżek absolutnych
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_TB_LOG_DIR = os.path.join(PROJECT_ROOT, "runs", "micro_world_ppo")
DEFAULT_CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


# ---------- Env factory ----------
def make_env(seed=123, rank=0):
    def _thunk():
        from src.micro_world.env_gym import MicroWorldVisionEnv
        env = MicroWorldVisionEnv(seed=seed + rank, max_steps=5000, render_mode=None)
        return env

    return _thunk


# ---------- Train ----------
def train(timesteps: int, tb_logdir: str, ckpt_dir: str, seed: int):
    num_cpu = 6
    print(f"Using {num_cpu} CPU processes for parallel training.")

    env = SubprocVecEnv([make_env(seed=seed, rank=i) for i in range(num_cpu)])
    env = VecMonitor(env)

    eval_env = DummyVecEnv([make_env(seed=seed + 1000)])
    eval_env = VecMonitor(eval_env)

    policy_kwargs = {
        "features_extractor_class": SmallCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": dict(pi=[256, 128], vf=[256, 128]),
    }

    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=8192,

        # --- ZMIANY TUTAJ ---
        batch_size=512,  # Było 256. Dajemy GPU większe porcje.
        n_epochs=10,  # Było 4. GPU trenuje dłużej na tych samych danych.
        # --- KONIEC ZMIAN ---

        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.03,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tb_logdir,
        seed=seed,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )

    new_logger = configure(tb_logdir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    print(f"Ensuring checkpoint directory exists at: {ckpt_dir}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Poprawne obliczanie częstotliwości (zgodnie z Twoją sugestią)
    save_freq_total = 50_000
    save_freq_per_env = max(1, save_freq_total // num_cpu)  # 50_000 / 6 ≈ 8333

    eval_freq_total = 10_000
    eval_freq_per_env = max(1, eval_freq_total // num_cpu)  # 10_000 / 6 ≈ 1666

    print(f"Checkpoint save freq per env: {save_freq_per_env} (Total steps: ~{save_freq_per_env * num_cpu})")
    print(f"Evaluation freq per env: {eval_freq_per_env} (Total steps: ~{eval_freq_per_env * num_cpu})")

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_per_env,
        save_path=ckpt_dir,
        name_prefix="ppo_mw",
        verbose=1
    )

    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=ckpt_dir,
        log_path=ckpt_dir,
        eval_freq=eval_freq_per_env,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        callback_after_eval=stop_train_callback
    )

    print(f"\n{'=' * 60}")
    print(f"  Starting PPO Training")
    print(f"{'=' * 60}")
    print(f"Total timesteps:     {timesteps:,}")
    print(f"Entropy coef:        {model.ent_coef}")
    print(f"Tensorboard:         {tb_logdir}")
    print(f"Checkpoints:         {ckpt_dir}")
    print(f"Device:              {model.device}")
    print(f"{'=' * 60}\n")

    print("TIPS:")
    print("- Watch ep_rew_mean in tensorboard - should increase")
    print("- Watch eval/mean_ep_length - should increase above 1000")
    print()

    callback_list = [checkpoint_callback, eval_callback]

    model.learn(
        total_timesteps=timesteps,
        callback=callback_list,
        progress_bar=True
    )

    final_path = os.path.join(ckpt_dir, "ppo_mw_final.zip")
    model.save(final_path)
    print(f"\n✓ Training complete!")
    print(f"✓ Final model: {final_path}")
    print(f"✓ Best model:  {ckpt_dir}/best_model.zip")


# ---------- Play ----------
def play(model_path: str, episodes: int = 5, render_vis: bool = False):
    print(f"Loading model from: {model_path}")

    if render_vis:
        print("Visual rendering not implemented in play mode. Use viz_realtime.py instead:")
        print(f"  python -m src.scripts.viz_realtime --model {model_path}")
        return

    if not os.path.isabs(model_path) and not model_path.startswith("checkpoints"):
        model_path = os.path.join(DEFAULT_CKPT_DIR, model_path)

    from src.micro_world.env_gym import MicroWorldVisionEnv
    env = MicroWorldVisionEnv(render_mode="ansi")

    policy_kwargs = {
        "features_extractor_class": SmallCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": dict(pi=[256, 128], vf=[256, 128]),
    }
    model = PPO.load(model_path, policy_kwargs=policy_kwargs)

    total_rewards = []
    total_scores = []

    for ep in range(episodes):
        obs, info = env.reset()
        done, trunc = False, False
        ep_r = 0.0
        step = 0

        print(f"\n{'=' * 50}")
        print(f"Episode {ep + 1}/{episodes}")
        print(f"{'=' * 50}")

        while not (done or trunc) and step < 5000:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            ep_r += r
            step += 1

            if step % 500 == 0:
                print(f"  Step {step:4d}: E={info['energy']:.2f}, score={info['score']:.0f}, reward_sum={ep_r:.2f}")

        total_rewards.append(ep_r)
        total_scores.append(info['score'])

        print(f"\n✓ Episode {ep + 1} results:")
        print(f"  Total reward:    {ep_r:.3f}")
        print(f"  Steps survived:  {step}")
        print(f"  Final energy:    {info['energy']:.2f}")
        print(f"  Final score:     {info['score']:.0f}")
        print(f"  Terminated:      {done}")
        print(f"  Truncated:       {trunc}")

    env.close()

    print(f"\n{'=' * 50}")
    print(f"Summary over {episodes} episodes:")
    print(f"{'=' * 50}")
    print(f"  Avg reward:  {sum(total_rewards) / len(total_rewards):.3f}")
    print(f"  Avg score:   {sum(total_scores) / len(total_scores):.1f}")
    print(f"  Max score:   {max(total_scores):.0f}")
    print(f"  Min score:   {min(total_scores):.0f}")


# ---------- CLI ----------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps (500k = ~30min, 2M = ~2h)")
    parser.add_argument("--tb", type=str, default=DEFAULT_TB_LOG_DIR)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT_DIR)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--play", type=str, default=None, help="Path to .zip model to test")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes")
    args = parser.parse_args()

    if args.play:
        play(args.play, episodes=args.episodes)
    else:
        train(args.timesteps, args.tb, args.ckpt, args.seed)