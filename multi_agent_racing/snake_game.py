import random, collections
import numpy as np
import pygame

WIDTH_CELLS = 25
HEIGHT_CELLS = 14
CELL = 30
WIDTH_PX = WIDTH_CELLS * CELL
HEIGHT_PX = HEIGHT_CELLS * CELL
FPS = 30

# Ajan renkleri: (Body Outer, Body Inner, Head Outer, Head Inner)
AGENT_COLORS = [
    ((0, 0, 255), (0, 100, 255), (0, 50, 150), (0, 200, 255)),       # Mavi
    ((128, 0, 128), (180, 50, 180), (75, 0, 130), (220, 100, 220)),  # Mor
    ((255, 140, 0), (255, 180, 50), (200, 100, 0), (255, 220, 100)), # Turuncu
    ((0, 200, 200), (50, 255, 255), (0, 120, 120), (150, 255, 255))  # Cyan
]

class SnakeEnv:
    DIRS = [(0,-1),(1,0),(0,1),(-1,0)]

    def __init__(self, agent_idx=0):
        self.agent_idx = agent_idx
        self.body_out, self.body_in, self.head_out, self.head_in = AGENT_COLORS[agent_idx % 4]
        self._all_cells = set((x, y) for x in range(WIDTH_CELLS) for y in range(HEIGHT_CELLS))
        self.rng = random.Random() # Food için özel random generator

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
            
        cx, cy = WIDTH_CELLS // 2, HEIGHT_CELLS // 2
        self.snake = collections.deque([(cx,cy),(cx-1,cy),(cx-2,cy)])
        self.snake_set = {(cx,cy),(cx-1,cy),(cx-2,cy)}
        self.dir_i = 1
        self.score = 0
        self.steps = 0
        self.max_steps = WIDTH_CELLS * HEIGHT_CELLS * 2
        self.done = False
        self._place_food()
        return self._obs()

    def _place_food(self):
        empty = self._all_cells - self.snake_set
        # self.rng.choice list istiyor, o yüzden tuple yapıp sortluyoruz ki seed deterministik kalsın
        # Python'da set tuple yapıldığında sıra garanti değildir, o yüzden sıralamak önemli.
        empty_sorted = sorted(list(empty)) 
        self.food = self.rng.choice(empty_sorted) if empty_sorted else (0, 0)

    def step(self, action):
        if self.done:
            return self._obs(), 0.0, True
            
        self.steps += 1
        self.dir_i = (self.dir_i + (1 if action==1 else -1 if action==2 else 0)) % 4
        dx, dy = self.DIRS[self.dir_i]
        hx, hy = self.snake[0]
        nx, ny = hx+dx, hy+dy

        if nx<0 or nx>=WIDTH_CELLS or ny<0 or ny>=HEIGHT_CELLS or (nx,ny) in self.snake_set:
            self.done = True
            return self._obs(), -10.0, True

        head = (nx, ny)
        self.snake.appendleft(head)
        self.snake_set.add(head)

        if head == self.food:
            self.score += 1
            reward = 10.0
            self._place_food()
        else:
            tail = self.snake.pop()
            self.snake_set.discard(tail)
            reward = -0.01

        if self.steps >= self.max_steps:
            self.done = True
            
        return self._obs(), reward, self.done

    def _obs(self):
        hx, hy = self.snake[0]
        dx, dy = self.DIRS[self.dir_i]
        lx, ly =  dy, -dx
        rx, ry = -dy,  dx

        def danger(ox, oy):
            nx, ny = hx+ox, hy+oy
            return float(nx<0 or nx>=WIDTH_CELLS or ny<0 or ny>=HEIGHT_CELLS or (nx,ny) in self.snake_set)

        fx, fy = self.food
        return np.array([
            danger(dx,dy), danger(lx,ly), danger(rx,ry),
            float(dx== 1), float(dx==-1),
            float(dy==-1), float(dy== 1),
            float(fx>hx),  float(fx<hx),
            float(fy<hy),  float(fy>hy),
        ], dtype=np.float32)

    def draw(self, surface, is_winner=False, text_font=None):
        # Checkerboard yeşil zemin
        color1 = (170, 215, 81) # LIGHT_GREEN
        color2 = (162, 209, 73) # DARK_GREEN
        
        for y in range(HEIGHT_CELLS):
            for x in range(WIDTH_CELLS):
                rect = (x*CELL, y*CELL, CELL, CELL)
                color = color1 if (x+y)%2 == 0 else color2
                pygame.draw.rect(surface, color, rect)
                
        # Food (Elma - kırmızı kare)
        fx, fy = self.food
        pygame.draw.rect(surface, (255, 0, 0), (fx*CELL, fy*CELL, CELL, CELL))

        # Snake
        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                c_out, c_in = self.head_out, self.head_in
            else:
                c_out, c_in = self.body_out, self.body_in
                
            pygame.draw.rect(surface, c_out, (x*CELL, y*CELL, CELL, CELL))
            pygame.draw.rect(surface, c_in, (x*CELL + 4, y*CELL + 4, CELL - 8, CELL - 8))
                
        # Skor
        if text_font:
            txt = text_font.render(f"Skor: {self.score}", True, (0, 0, 0)) # Siyah yazı
            surface.blit(txt, (6, 4))

        # Eğer ölmüşse karart
        if self.done and not is_winner:
            s = pygame.Surface((WIDTH_PX, HEIGHT_PX), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180)) # Yarı saydam siyah
            surface.blit(s, (0,0))
            
        # Eğer kazanmışsa
        if is_winner and text_font:
            win_txt = text_font.render(f"Kazanan: Ajan {self.agent_idx + 1}", True, (255, 215, 0))
            win_rect = win_txt.get_rect(center=(WIDTH_PX//2, HEIGHT_PX//2))
            # Gölge / arkaplan
            pygame.draw.rect(surface, (0,0,0), win_rect.inflate(20, 20))
            surface.blit(win_txt, win_rect)

class MultiSnakeEnv:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.envs = [SnakeEnv(i) for i in range(4)]
        self.screen = None
        self.clock = None
        self.font = None
        
        # Ekran boyutları: 2x2 grid
        self.screen_w = WIDTH_PX * 2
        self.screen_h = HEIGHT_PX * 2
        
        if render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
            pygame.display.set_caption("Multi-Agent Snake RL Racing")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("monospace", 22, bold=True)
            
        self.surfaces = [pygame.Surface((WIDTH_PX, HEIGHT_PX)) for _ in range(4)]
        self.positions = [(0,0), (WIDTH_PX, 0), (0, HEIGHT_PX), (WIDTH_PX, HEIGHT_PX)]

    def reset(self):
        seed = random.randint(0, 999999) # Aynı yiyecek dizilimi için ortak seed
        return [env.reset(seed) for env in self.envs]
        
    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        obs = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        return obs, rewards, dones
        
    def render(self, is_race=False):
        if not self.render_mode:
            return
            
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.close()
                exit()
                
        self.screen.fill((50,50,50)) # Arka plan çerçevesi
        
        # Hayatta kalan ajan sayısını bul (yarışma modu için)
        active_agents = [i for i, env in enumerate(self.envs) if not env.done]
        winner_idx = active_agents[0] if is_race and len(active_agents) == 1 else -1
        if is_race and len(active_agents) == 0: # Hepsi öldüyse, en son öleni winner yapamayız gerçi, neyse.
            pass

        for i, env in enumerate(self.envs):
            surf = self.surfaces[i]
            is_winner = (i == winner_idx) and is_race
            env.draw(surf, is_winner=is_winner, text_font=self.font)
            
            # Sınır çizgisi
            pygame.draw.rect(surf, (200,200,200), surf.get_rect(), 2)
            
            # Ana ekrana blit
            self.screen.blit(surf, self.positions[i])
            
        pygame.display.flip()
        self.clock.tick(FPS)
        
    def close(self):
        if self.render_mode:
            pygame.quit()
