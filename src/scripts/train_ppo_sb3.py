# src/scripts/train_ppo_sb3.py
# Train RecurrentPPO or MaskablePPO on MicroWorldVisionEnv

import argparse, os, numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import configure

from src.micro_world.env_gym import MicroWorldVisionEnv
from src.scripts.custom_policy import SmallCNN
from src.scripts.micro_world_vision import ACTIONS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_TB_LOG_DIR = os.path.join(PROJECT_ROOT, "runs", "micro_world_ppo")
DEFAULT_CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")


def mask_fn(env: MicroWorldVisionEnv):
    """Return boolean mask of valid actions at the current state."""
    y, x = env.agent.y, env.agent.x
    mask = np.ones(len(ACTIONS), dtype=bool)
    for i, (dy, dx) in enumerate(ACTIONS):
        ny, nx = y + dy, x + dx
        if not (0 <= ny < env.world.h and 0 <= nx < env.world.w):
            mask[i] = False
        elif env.world.W[ny, nx] > 0:
            mask[i] = False
    return mask


def make_env(seed=123, rank=0, use_masking=False):
    def _thunk():
        env = MicroWorldVisionEnv(seed=seed + rank, max_steps=5000, render_mode=None)
        if use_masking:
            env = ActionMasker(env, mask_fn)
        return env

    return _thunk


def train(timesteps: int, tb_logdir: str, ckpt_dir: str, seed: int,
          use_recurrent: bool = True, use_masking: bool = False):
    """
    Train either RecurrentPPO or MaskablePPO

    Args:
        timesteps: Total timesteps to train
        tb_logdir: TensorBoard log directory
        ckpt_dir: Checkpoint directory
        seed: Random seed
        use_recurrent: If True, use RecurrentPPO; if False, use MaskablePPO
        use_masking: If True, use action masking (only for MaskablePPO)
    """
    n_envs = 6
    env = SubprocVecEnv([make_env(seed=seed, rank=i, use_masking=use_masking) for i in range(n_envs)])
    env = VecMonitor(env)

    eval_env = DummyVecEnv([make_env(seed=seed + 1000, use_masking=use_masking)])
    eval_env = VecMonitor(eval_env)

    # Policy kwargs for RecurrentPPO
    if use_recurrent:
        policy_kwargs = dict(
            features_extractor_class=SmallCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            enable_critic_lstm=True,  # LSTM dla krytyka
            lstm_hidden_size=256,  # Rozmiar LSTM
        )

        model = RecurrentPPO(
            policy="CnnLstmPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=tb_logdir,
            seed=seed,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
        model_name = "recurrent_ppo"
    else:
        # Policy kwargs for MaskablePPO (bez LSTM)
        policy_kwargs = dict(
            features_extractor_class=SmallCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 128], vf=[256, 128])
        )

        model = MaskablePPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=tb_logdir,
            seed=seed,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )
        model_name = "maskable_ppo" if use_masking else "ppo"

    new_logger = configure(tb_logdir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    os.makedirs(ckpt_dir, exist_ok=True)
    save_freq_total = 50_000
    save_freq_per_env = max(1, save_freq_total // n_envs)
    eval_freq_total = 10_000
    eval_freq_per_env = max(1, eval_freq_total // n_envs)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_per_env,
        save_path=ckpt_dir,
        name_prefix=f"{model_name}_mw",
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
        callback_after_eval=stop_train_callback,
    )

    print(f"Starting {model_name.upper()} training...")
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    final_path = os.path.join(ckpt_dir, f"{model_name}_mw_final.zip")
    model.save(final_path)
    print(f"✓ Training complete! Final model: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--tb", type=str, default=DEFAULT_TB_LOG_DIR)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--recurrent", action="store_true",
                        help="Use RecurrentPPO instead of MaskablePPO")
    parser.add_argument("--no-masking", action="store_true",
                        help="Disable action masking (only for non-recurrent)")
    args = parser.parse_args()

    use_masking = not args.no_masking and not args.recurrent

    train(
        args.timesteps,
        args.tb,
        args.ckpt,
        args.seed,
        use_recurrent=args.recurrent,
        use_masking=use_masking
    )