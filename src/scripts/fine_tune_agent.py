# fine_tune_agent.py
# Fine-tuning module for continued training from saved checkpoints
# Allows incremental improvement and adaptation to new scenarios

import argparse
import os
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from sb3_contrib.common.wrappers import ActionMasker

from src.micro_world.env_gym import MicroWorldVisionEnv
from custom_policy import SmallCNN
from micro_world_vision import ACTIONS


def mask_fn(env: MicroWorldVisionEnv):
    """Return boolean mask of valid actions"""
    import numpy as np
    y, x = env.agent.y, env.agent.x
    mask = np.ones(len(ACTIONS), dtype=bool)
    for i, (dy, dx) in enumerate(ACTIONS):
        ny, nx = y + dy, x + dx
        if not (0 <= ny < env.world.h and 0 <= nx < env.world.w):
            mask[i] = False
        elif env.world.W[ny, nx] > 0:
            mask[i] = False
    return mask


def make_env(seed=123, rank=0, use_masking=False, n_pellets=500, harder=False):
    def _thunk():
        env = MicroWorldVisionEnv(
            seed=seed + rank,
            max_steps=8000 if harder else 5000,
            render_mode=None,
            n_pellets=n_pellets
        )
        if use_masking:
            env = ActionMasker(env, mask_fn)
        return env

    return _thunk


def detect_model_type(model_path):
    """Try to detect what type of model this is"""
    import zipfile
    try:
        with zipfile.ZipFile(model_path, 'r') as z:
            # Check for LSTM-related keys
            data = z.read('data')
            if b'lstm' in data.lower():
                return 'recurrent'
            elif b'action_mask' in data.lower():
                return 'maskable'
            else:
                return 'ppo'
    except:
        # Default to maskable as it's most common
        return 'maskable'


def fine_tune(
        model_path: str,
        timesteps: int,
        tb_logdir: str,
        ckpt_dir: str,
        seed: int,
        learning_rate: float = 1e-4,  # Lower LR for fine-tuning
        harder: bool = False,
        model_type: str = None
):
    """
    Fine-tune an existing model with optionally harder scenarios

    Args:
        model_path: Path to existing model checkpoint
        timesteps: Additional timesteps to train
        tb_logdir: TensorBoard log directory
        ckpt_dir: Checkpoint save directory
        seed: Random seed
        learning_rate: Learning rate (lower than initial training)
        harder: If True, use harder environment settings
        model_type: Force specific model type ('ppo', 'maskable', 'recurrent')
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\n🔧 FINE-TUNING MODE")
    print(f"   Loading model from: {model_path}")
    print(f"   Additional timesteps: {timesteps:,}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Harder scenarios: {harder}")

    # Auto-detect model type if not specified
    if model_type is None:
        model_type = detect_model_type(model_path)
        print(f"   Detected model type: {model_type}")

    use_masking = model_type == 'maskable'
    use_recurrent = model_type == 'recurrent'

    # Create environments
    n_envs = 4  # Fewer envs for fine-tuning
    n_pellets = 400 if harder else 500

    env = SubprocVecEnv([
        make_env(seed=seed, rank=i, use_masking=use_masking,
                 n_pellets=n_pellets, harder=harder)
        for i in range(n_envs)
    ])
    env = VecMonitor(env)

    eval_env = DummyVecEnv([
        make_env(seed=seed + 1000, use_masking=use_masking,
                 n_pellets=n_pellets, harder=harder)
    ])
    eval_env = VecMonitor(eval_env)

    # Policy kwargs
    if use_recurrent:
        policy_kwargs = dict(
            features_extractor_class=SmallCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            enable_critic_lstm=True,
            lstm_hidden_size=256,
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=SmallCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 128], vf=[256, 128])
        )

    # Load existing model
    print(f"\n📦 Loading model...")
    if use_recurrent:
        model = RecurrentPPO.load(model_path, env=env, policy_kwargs=policy_kwargs)
        model_name = "recurrent_ppo_finetuned"
    elif use_masking:
        model = MaskablePPO.load(model_path, env=env, policy_kwargs=policy_kwargs)
        model_name = "maskable_ppo_finetuned"
    else:
        model = PPO.load(model_path, env=env, policy_kwargs=policy_kwargs)
        model_name = "ppo_finetuned"

    print(f"✓ Model loaded successfully")

    # Adjust hyperparameters for fine-tuning
    model.learning_rate = learning_rate
    model.n_steps = 2048
    model.batch_size = 512
    model.n_epochs = 8  # Slightly fewer epochs
    model.ent_coef = 0.01  # Lower exploration

    print(f"\n⚙ Fine-tuning hyperparameters:")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Steps per update: {model.n_steps}")
    print(f"   Batch size: {model.batch_size}")
    print(f"   Epochs: {model.n_epochs}")
    print(f"   Entropy coefficient: {model.ent_coef}")

    # Setup logging
    tb_log_name = f"{model_name}_{'harder' if harder else 'standard'}"
    new_logger = configure(os.path.join(tb_logdir, tb_log_name),
                           ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # Callbacks
    os.makedirs(ckpt_dir, exist_ok=True)

    save_freq_total = 25_000
    save_freq_per_env = max(1, save_freq_total // n_envs)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_per_env,
        save_path=ckpt_dir,
        name_prefix=model_name,
        verbose=1
    )

    eval_freq_total = 10_000
    eval_freq_per_env = max(1, eval_freq_total // n_envs)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(ckpt_dir, "best_finetuned"),
        log_path=ckpt_dir,
        eval_freq=eval_freq_per_env,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    # Train (fine-tune)
    print(f"\n🚀 Starting fine-tuning...")
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        reset_num_timesteps=False  # Continue from previous timesteps
    )

    # Save final fine-tuned model
    final_path = os.path.join(ckpt_dir, f"{model_name}_final.zip")
    model.save(final_path)

    print(f"\n✅ Fine-tuning complete!")
    print(f"   Final model saved to: {final_path}")
    print(f"   Best model saved to: {os.path.join(ckpt_dir, 'best_finetuned')}")


def compare_models(original_path: str, finetuned_path: str, episodes: int = 10):
    """
    Compare performance of original vs fine-tuned model
    """
    from micro_world_vision import World, Agent
    import numpy as np

    print(f"\n📊 COMPARING MODELS")
    print(f"   Original: {original_path}")
    print(f"   Fine-tuned: {finetuned_path}")
    print(f"   Episodes: {episodes}")

    # Load models
    try:
        from custom_policy import SmallCNN
        policy_kwargs = {
            "features_extractor_class": SmallCNN,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": {"pi": [256, 128], "vf": [256, 128]}
        }

        try:
            original = MaskablePPO.load(original_path, policy_kwargs=policy_kwargs)
            finetuned = MaskablePPO.load(finetuned_path, policy_kwargs=policy_kwargs)
            model_type = "maskable"
        except:
            original = PPO.load(original_path, policy_kwargs=policy_kwargs)
            finetuned = PPO.load(finetuned_path, policy_kwargs=policy_kwargs)
            model_type = "ppo"
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return

    def evaluate_model(model, name):
        scores = []
        survivals = []
        levels = []

        for ep in range(episodes):
            world = World(seed=1000 + ep, n_pellets=500)
            agent = Agent(world, agent_id=0)

            steps = 0
            max_steps = 5000

            while agent.energy > 0 and steps < max_steps:
                world.step_fields()
                obs = agent.get_obs_img().astype(np.float32)
                obs_hwc = np.clip(obs, 0.0, 1.0).transpose(1, 2, 0)
                obs_hwc = (obs_hwc * 255).astype(np.uint8)

                if model_type == "maskable":
                    mask = np.ones(len(ACTIONS), dtype=bool)
                    action, _ = model.predict(obs_hwc, action_masks=mask,
                                              deterministic=True)
                else:
                    action, _ = model.predict(obs_hwc, deterministic=True)

                agent.step(int(action))
                steps += 1

            scores.append(agent.score)
            survivals.append(steps)
            levels.append(agent.level)

        return {
            'avg_score': np.mean(scores),
            'std_score': np.std(scores),
            'avg_survival': np.mean(survivals),
            'std_survival': np.std(survivals),
            'avg_level': np.mean(levels),
            'max_level': np.max(levels),
        }

    print("\n⏳ Evaluating original model...")
    original_stats = evaluate_model(original, "Original")

    print("⏳ Evaluating fine-tuned model...")
    finetuned_stats = evaluate_model(finetuned, "Fine-tuned")

    # Display comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    metrics = [
        ("Average Score", "avg_score"),
        ("Score Std Dev", "std_score"),
        ("Average Survival", "avg_survival"),
        ("Survival Std Dev", "std_survival"),
        ("Average Level", "avg_level"),
        ("Max Level Reached", "max_level"),
    ]

    print(f"\n{'Metric':<25} {'Original':<15} {'Fine-tuned':<15} {'Change':<15}")
    print("-" * 70)

    for label, key in metrics:
        orig_val = original_stats[key]
        fine_val = finetuned_stats[key]
        change = fine_val - orig_val
        change_pct = (change / orig_val * 100) if orig_val != 0 else 0

        change_str = f"{change:+.2f} ({change_pct:+.1f}%)"

        print(f"{label:<25} {orig_val:<15.2f} {fine_val:<15.2f} {change_str:<15}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a trained agent")

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Fine-tune command
    finetune_parser = subparsers.add_parser('train', help='Fine-tune a model')
    finetune_parser.add_argument("--model", type=str, required=True,
                                 help="Path to model to fine-tune")
    finetune_parser.add_argument("--timesteps", type=int, default=100_000,
                                 help="Additional training timesteps")
    finetune_parser.add_argument("--tb", type=str, default="./runs/finetuning",
                                 help="TensorBoard log directory")
    finetune_parser.add_argument("--ckpt", type=str, default="./checkpoints",
                                 help="Checkpoint directory")
    finetune_parser.add_argument("--seed", type=int, default=456,
                                 help="Random seed")
    finetune_parser.add_argument("--lr", type=float, default=1e-4,
                                 help="Learning rate for fine-tuning")
    finetune_parser.add_argument("--harder", action="store_true",
                                 help="Use harder environment settings")
    finetune_parser.add_argument("--type", type=str, choices=['ppo', 'maskable', 'recurrent'],
                                 help="Force specific model type")

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare models')
    compare_parser.add_argument("--original", type=str, required=True,
                                help="Path to original model")
    compare_parser.add_argument("--finetuned", type=str, required=True,
                                help="Path to fine-tuned model")
    compare_parser.add_argument("--episodes", type=int, default=10,
                                help="Number of episodes to evaluate")

    args = parser.parse_args()

    if args.command == 'train':
        fine_tune(
            model_path=args.model,
            timesteps=args.timesteps,
            tb_logdir=args.tb,
            ckpt_dir=args.ckpt,
            seed=args.seed,
            learning_rate=args.lr,
            harder=args.harder,
            model_type=args.type
        )
    elif args.command == 'compare':
        compare_models(args.original, args.finetuned, args.episodes)
    else:
        parser.print_help()