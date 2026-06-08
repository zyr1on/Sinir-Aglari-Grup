"""
    python train.py --episodes 2000
    python train.py --episodes 1000 --agents dqn ddqn
    python train.py --episodes 2000 --agents dqn ddqn rainbow
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

from snake_env import SnakeEnv
from agent import DQNAgent, DDQNAgent, RainbowAgent

GRID_SIZE   = 20
EPS_START   = 1.0
EPS_END     = 0.05
PRINT_EVERY = 50

MODELS_DIR      = "models"
MODELS_BEST_DIR = "models_best"  # En iyi modeller için yeni klasör sabiti


def calc_eps_decay(episodes: int, eps_start=EPS_START, eps_end=EPS_END, target_frac=0.80):
    return (eps_end / eps_start) ** (1.0 / (target_frac * max(1, episodes)))


def run_episode(agent, env):
    state = env.reset()
    total_reward = 0.0
    while True:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.train_step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            return info["score"], total_reward


def print_header():
    print(f"\n{'Agent':<12} {'Ep':>6} {'Score':>7} {'Avg100':>8} {'ε':>7}  {'Time':>6}")
    print("─" * 55)


def print_row(name, ep, score, avg100, eps_str, elapsed, suffix=""):
    t = f"{elapsed:.0f}s" if elapsed < 60 else f"{elapsed/60:.1f}m"
    line = f"{name:<12} {ep:>6} {score:>7} {avg100:>8.2f} {eps_str:>7}  {t:>6}"
    if suffix:
        line += f"  {suffix}"
    print(line)


def train_agent(agent, name, episodes, env_kwargs, key, models_dir, models_best_dir):
    env = SnakeEnv(**env_kwargs)
    scores     = []
    avg_scores = []
    start      = time.time()
    last_print = time.time()

    best_score = -1          
    first_block = True

    for ep in range(1, episodes + 1):
        score, _ = run_episode(agent, env)
        agent.decay_epsilon()
        scores.append(score)

        # en iyi skor 
        if score > best_score:
            best_score = score
            # En iyi modeli models_best_dir içine kaydet
            best_path  = os.path.join(models_best_dir, f"{key}_best.pt")
            agent.save(best_path)

            if first_block:
                print_header()
                first_block = False

            avg100 = float(np.mean(scores[-100:]))
            elapsed = time.time() - last_print
            print_row(
                name, ep, score, avg100, agent.epsilon_str, elapsed,
                suffix=f"(en iyi skor: {best_score}, best model kaydedildi)"
            )
            sys.stdout.flush()

        if ep % PRINT_EVERY == 0:
            avg100 = float(np.mean(scores[-100:]))
            elapsed = time.time() - last_print
            last_print = time.time()

            if first_block:
                print_header()
                first_block = False

            print_row(name, ep, score, avg100, agent.epsilon_str, elapsed)
            sys.stdout.flush()
            avg_scores.append(avg100)

    total = time.time() - start
    print(f"\n[{name}] Training done — {episodes} eps in {total:.1f}s  "
          f"| Best avg100: {max(avg_scores, default=0):.2f}")
    return scores


def main():
    parser = argparse.ArgumentParser(description="Snake RL Training")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--agents",   nargs="+",
                        choices=["dqn","ddqn","rainbow"],
                        default=["dqn","ddqn","rainbow"])
    parser.add_argument("--grid",     type=int, default=GRID_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n[Device] Using: {'cuda' if torch.cuda.is_available() else 'cpu'} ({gpu_name})")

    # Klasörleri oluştur
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(MODELS_BEST_DIR, exist_ok=True)

    eps_decay = calc_eps_decay(args.episodes)
    print(f"[Config] Episodes={args.episodes}  eps_decay={eps_decay:.5f}  grid={args.grid}×{args.grid}")

    env_kwargs = {"grid_size": args.grid}
    obs_size   = SnakeEnv.OBS_SIZE
    n_actions  = SnakeEnv.N_ACTIONS

    agent_map = {
        "dqn":     ("DQN",     DQNAgent(obs_size,  n_actions, device, eps_decay=eps_decay)),
        "ddqn":    ("DDQN",    DDQNAgent(obs_size, n_actions, device, eps_decay=eps_decay)),
        "rainbow": ("Rainbow", RainbowAgent(obs_size, n_actions, device)),
    }

    for key in args.agents:
        label, agent = agent_map[key]
        print(f"\n{'='*55}")
        print(f"  Training: {label}  ({args.episodes} episodes)")
        print(f"{'='*55}")
        
        # Güncellenmiş fonksiyon çağrısı
        train_agent(agent, label, args.episodes, env_kwargs, key, MODELS_DIR, MODELS_BEST_DIR)
        
        # Final modeli normal models/ klasörüne kaydedilir
        save_path = os.path.join(MODELS_DIR, f"{key}.pt")
        agent.save(save_path)
        print(f"[{label}] final model saved -> {save_path}")

    print("\n[Done] All agents trained. Run `python play.py`.")


if __name__ == "__main__":
    main()