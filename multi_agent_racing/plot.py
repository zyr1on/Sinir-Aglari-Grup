import json
import matplotlib.pyplot as plt

def plot_scores(json_path="scores.json"):
    try:
        with open(json_path, "r") as f:
            all_scores = json.load(f)
    except FileNotFoundError:
        print(f"Dosya bulunamadı: {json_path}")
        return

    colors = ['c', 'y', 'm', 'k'] # Cyan, Yellow, Magenta, Black/Gray
    labels = ['Ajan 1 (Hızlı)', 'Ajan 2 (Yavaş)', 'Ajan 3 (Greedy)', 'Ajan 4 (Standart)']

    plt.figure(figsize=(10, 6))
    
    # Hareketli ortalama hesaplama (daha pürüzsüz grafik için)
    def moving_average(data, window_size=50):
        return [sum(data[max(0, i-window_size):i+1]) / min(i+1, window_size) for i in range(len(data))]

    for i in range(4):
        agent_key = f"Agent_{i+1}"
        if agent_key in all_scores:
            scores = all_scores[agent_key]
            smoothed_scores = moving_average(scores, window_size=50)
            
            # Arka planda silik olarak gerçek skorlar
            plt.plot(scores, color=colors[i], alpha=0.2)
            # Öne çıkan hareketli ortalama
            plt.plot(smoothed_scores, color=colors[i], label=labels[i], linewidth=2)

    plt.title("4 Farklı Ajanın Eğitim Süreci")
    plt.xlabel("Bölüm (Episode)")
    plt.ylabel("Skor")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()

if __name__ == "__main__":
    plot_scores()
