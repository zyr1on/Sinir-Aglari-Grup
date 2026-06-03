import argparse
import json
from snake_game import MultiSnakeEnv
from model import DQNAgent

def train(episodes=2000, render=False):
    env = MultiSnakeEnv(render_mode=render)
    
    # 4 Farklı Ajan
    agents = [
        DQNAgent(lr=0.002,  eps_decay=0.990, gamma=0.95), # Ajan 1: Hızlı öğrenen (Fast Learner)
        DQNAgent(lr=0.0005, eps_decay=0.998, gamma=0.95), # Ajan 2: Yavaş öğrenen (Slow Learner)
        DQNAgent(lr=0.001,  eps_decay=0.980, gamma=0.99), # Ajan 3: Açgözlü/İleri görüşlü (Greedy)
        DQNAgent(lr=0.001,  eps_decay=0.995, gamma=0.95)  # Ajan 4: Standart
    ]
    
    best_scores = [0, 0, 0, 0]
    all_scores = {f"Agent_{i+1}": [] for i in range(4)}

    for ep in range(1, episodes + 1):
        obs_list = env.reset()
        dones = [False, False, False, False]
        total_rewards = [0.0, 0.0, 0.0, 0.0]
        
        while not all(dones): # Bütün ajanlar ölene kadar bekle
            if render: 
                env.render()
                
            actions = []
            for i, agent in enumerate(agents):
                if not dones[i]:
                    actions.append(agent.act(obs_list[i]))
                else:
                    actions.append(0) # Ölmüşse dummy action
                    
            nobs_list, rewards, new_dones = env.step(actions)
            
            for i, agent in enumerate(agents):
                if not dones[i]:
                    agent.store(obs_list[i], actions[i], rewards[i], nobs_list[i], new_dones[i])
                    if agent.step_n % 4 == 0:
                        agent.learn()
                        
            obs_list = nobs_list
            for i in range(4):
                if not dones[i]:
                    total_rewards[i] += rewards[i]
            dones = new_dones

        # Episode bitti, epsilonları güncelle ve skorları kaydet
        scores_str = ""
        for i, agent in enumerate(agents):
            agent.decay_epsilon()
            score = env.envs[i].score
            all_scores[f"Agent_{i+1}"].append(score)
            
            if score > best_scores[i]:
                best_scores[i] = score
                agent.save(f"agent_{i+1}_best.pt")
                
            scores_str += f"A{i+1}:{score:2d} "

        if ep % 50 == 0 or ep == 1:
            print(f"[{ep:5d}/{episodes}] Skorlar: {scores_str} | En İyiler: {best_scores}")

        # Her 100 bölümde bir skorları dosyaya yazdır
        if ep % 100 == 0:
            with open("scores.json", "w") as f:
                json.dump(all_scores, f)
                
    # Eğitim sonu
    with open("scores.json", "w") as f:
        json.dump(all_scores, f)
        
    for i, agent in enumerate(agents):
        agent.save(f"agent_{i+1}_final.pt")
        
    env.close()
    print("Eğitim tamamlandı! Skorlar scores.json dosyasına kaydedildi.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eps", type=int, default=1000)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()
    train(episodes=args.eps, render=args.render)
