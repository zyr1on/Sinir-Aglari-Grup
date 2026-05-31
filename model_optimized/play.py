"""
python play.py               # 5 oyun oynar
python play.py --games 10    # 10 oyun
python play.py --model best.pt
"""
import argparse, os
from snake_game import SnakeEnv
from model import DQNAgent

def play(path="snake_dqn.pt", games=5):
    if not os.path.exists(path):
        print(f"Model bulunamadı: {path}\nÖnce eğitin: python train.py")
        return

    env   = SnakeEnv(render_mode=True)
    agent = DQNAgent()
    agent.load(path)
    agent.policy.eval()   # ✅ inference modu
    agent.eps = 0.0       # ✅ kesinlikle greedy

    scores = []
    for g in range(1, games+1):
        obs  = env.reset()
        done = False
        while not done:
            env.render()
            action       = agent.act(obs, greedy=True)
            obs, _, done = env.step(action)
        scores.append(env.score)
        print(f"Oyun {g}: skor = {env.score}")

    env.close()
    print(f"\nOrtalama skor: {sum(scores)/len(scores):.1f}  "
          f"| En iyi: {max(scores)}  | En kötü: {min(scores)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="snake_dqn.pt")
    p.add_argument("--games", type=int, default=5)
    args = p.parse_args()
    play(path=args.model, games=args.games)
