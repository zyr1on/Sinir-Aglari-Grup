"""
python train.py              # 2000 ep, ekransız
python train.py --eps 5000   # daha uzun eğitim
python train.py --render     # eğitimi izle (yavaş)
"""
import argparse
from snake_game import SnakeEnv
from model import DQNAgent

def train(episodes=2000, render=False, save_path="snake_dqn.pt"):
    env   = SnakeEnv(render_mode=render)
    agent = DQNAgent()
    best  = 0

    for ep in range(1, episodes + 1):
        obs  = env.reset()
        done = False
        total_r = 0.0

        while not done:
            if render: env.render()
            action        = agent.act(obs)
            nobs, r, done = env.step(action)
            agent.store(obs, action, r, nobs, done)

            if agent.step_n % 4 == 0:   # her 4 adımda bir öğren
                agent.learn()

            obs      = nobs
            total_r += r

        if env.score > best:
            best = env.score

        if ep % 100 == 0 and best > 0:  # 100 ep'de bir kaydet
            agent.save(save_path)
            print(f"[{ep:5d}/{episodes}]  score={env.score:3d}  "
                  f"eps={agent.eps:.3f}  reward={total_r:7.1f}  best={best}  → kaydedildi")
        elif ep % 50 == 0:
            print(f"[{ep:5d}/{episodes}]  score={env.score:3d}  "
                  f"eps={agent.eps:.3f}  reward={total_r:7.1f}  best={best}")

    agent.save(save_path)               # eğitim sonunda mutlaka kaydet
    env.close()
    print(f"\nEğitim tamamlandı. En iyi skor: {best}  →  {save_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eps",    type=int, default=2000)
    p.add_argument("--render", action="store_true")
    p.add_argument("--model",  type=str, default="snake_dqn.pt")
    args = p.parse_args()
    train(episodes=args.eps, render=args.render, save_path=args.model)
