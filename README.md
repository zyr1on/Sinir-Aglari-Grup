# Sinir-Aglari-Grup
Uygulamalı Sinir Ağları Grubu


Collabratos: <br>
032390072 Semih Özdemir<br>
032390068 Eren Boylu<br>
032390048 Arda Berat Kosor<br>


# Things To Do

<li> Geleneksel DQN yerine Double DQN veya Rainbow DQN yapısı tercih edilebilir </li><
<li> Egitim için Paralel Environment kullanılabilir </li>
<li> Eğitim hızı için komple envoirment GPU alınabilir </li>


## Multi Agent Fikirleri
<li> Prosedürel harita da olan agent </li>
<li> av-avcı ilişkisi olan agent </li>
<li> <s>aynı veya farklı harita da yarışan agentlar</s> (Eklendi: `multi_agent_racing` modülü) </li>

# iyileştirilebilir
1. Double DQN — model.py'da 2 satır değişiklik
2. Raycasting — snake_game.py'a raycast() metodu eklenip obs vektörünü genişletilebilir (Kendi kendini yeme probleminie bir çözüm).
3. Kuyruk uzunluğu obs'a ekle — tek satır, raycast ile birlikte yapılabilir.
4. Prioritized Experience Replay — ReplayBuffer'ı yerine belki eklenebilir. Nadiren olan ama önemli deneyimleri (yiyecek yeme, ölüm) daha sık örnekle
5. Dueling DQN — QNet mimarisini değiştir. Ağı "bu durumun değeri" ve "bu aksiyonun avantajı" olarak ikiye böleriz. Plateau'yu kırmak için iyi.
6. Reward shaping — yiyeceğe yaklaşınca küçük pozitif, uzaklaşınca küçük negatif ödül ekle seyrek ödülden iyidir
7. Daha uzun eğitim + learning rate scheduler — 2000 episode yetmeyebilir. optim.lr_scheduler ile lr'yi zamanla düşürmek
8. Daha büyük/derin ağ — obs genişleyince hidden=256 yetersiz kalabilir, 512'ye çıkılır veya 3. katman eklenir

---

## Yeni Özellik: Çoklu Ajan Yarışı (Multi-Agent Racing)
Projeye `multi_agent_racing` klasörü altında yeni ve gelişmiş bir yarışma modülü eklenmiştir:

* **Paralel Eğitim:** 4 farklı stratejiye sahip ajan (Fast Learner, Slow Learner, Greedy, Standard) aynı anda eğitilir (`python train.py --render --fps 120`). Ajanların öğrenmeleri birbirleriyle eş zamanlı ilerler.
* **Skora Dayalı Turnuva Modu:** Eğitilen ajanlar `python race.py --fps 15` komutuyla yarışırlar. Oyun bittiğinde en son hayatta kalan değil, **en yüksek skora ulaşan** kazanır. Varsayılan olarak 5 oyun üzerinden şampiyon belirlenir, beraberlik durumunda (tie-breaker) ek oyunlar oynanır.
* **Anti-Loop (Döngü Engelleme):** Ajanların yiyecek almadan oyalanıp oyunu uzatmalarını engellemek için, art arda 150 adım boyunca hiçbir yiyecek yenmezse ajan otomatik olarak elenir.
* **Gelişmiş Veri Görselleştirme:** `python plot.py` kullanılarak, 2x2'lik grafik ızgarasında her ajanın kendi öğrenme serüveni incelenebilir. Grafikler ajanların **anlık puanlarını** kendi renklerinde ve **öğrenme trendlerini** (son 50 oyunun hareketli ortalaması) net bir kırmızı çizgiyle gösterir.
