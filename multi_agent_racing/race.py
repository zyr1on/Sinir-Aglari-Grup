import argparse
import time
from snake_game import MultiSnakeEnv
from model import DQNAgent

def race(games=5):
    env = MultiSnakeEnv(render_mode=True)
    
    agents = [DQNAgent() for _ in range(4)]
    
    # Modelleri yükle
    try:
        for i in range(4):
            agents[i].load(f"agent_{i+1}_best.pt")
            agents[i].policy.eval() # Test modu
            agents[i].eps = 0.0 # Tamamen greedy
    except Exception as e:
        print(f"Modeller yüklenirken hata oluştu: {e}")
        print("Önce modelleri eğitmek için 'python train.py' çalıştırın.")
        return

    for g in range(1, games+1):
        obs_list = env.reset()
        dones = [False, False, False, False]
        
        while not all(dones):
            env.render(is_race=True) # Yarışma modunda render (karartma vs)
            time.sleep(0.05) # Yarışmayı izleyebilmek için biraz yavaşlatıyoruz
            
            actions = []
            for i, agent in enumerate(agents):
                if not dones[i]:
                    actions.append(agent.act(obs_list[i], greedy=True))
                else:
                    actions.append(0)
                    
            obs_list, _, new_dones = env.step(actions)
            dones = new_dones
            
        # Oyun bitince 2 saniye bekle
        print(f"Yarış {g} bitti!")
        env.render(is_race=True) # Kazanan yazısını ekranda göstermek için
        time.sleep(3)
        
    env.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=3)
    args = p.parse_args()
    race(games=args.games)
