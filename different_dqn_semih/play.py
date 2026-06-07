"""
    python play.py                  # loads from ./models/
    python play.py --best           # loads from ./models_best/

Controls:
    Q        — speed UP
    E        — speed DOWN
    SPACE    — pause / unpause
    ESC      — quit
"""

import os
import sys
import time
import argparse
import torch
import pygame
import numpy as np

from snake_env import SnakeEnv
from agent import DQNAgent, DDQNAgent, RainbowAgent

# ── Display settings ──────────────────────────────────────────────────
CELL        = 22
GRID_SIZE   = 20
PANEL_PAD   = 12
INFO_HEIGHT = 80
FPS_DEFAULT = 12
FPS_MIN     = 2
FPS_MAX     = 60
FPS_STEP    = 2

MODELS_DIR      = "models"
MODELS_BEST_DIR = "models_best"

# ── Colors ────────────────────────────────────────────────────────────
BG          = (15,  17,  26)
PANEL_BG    = (22,  26,  40)
GRID_LINE   = (30,  35,  55)
SNAKE_HEAD  = (80, 220, 120)
SNAKE_BODY  = (40, 160,  80)
FOOD_COLOR  = (255, 80,  80)
TEXT_COLOR  = (200, 210, 230)
TITLE_COLOR = (120, 190, 255)
DEAD_COLOR  = (180,  60,  60)
SCORE_GLOW  = (255, 220,  80)
WIN_FLASH   = (255, 215,   0)

AGENT_COLORS = {
    "DQN":     (100, 160, 255),
    "DDQN":    (120, 255, 180),
    "Rainbow": (255, 160,  80),
}

PANEL_W = GRID_SIZE * CELL
PANEL_H = GRID_SIZE * CELL
WIN_W   = 3 * PANEL_W + 4 * PANEL_PAD
WIN_H   = PANEL_H + 2 * INFO_HEIGHT + 2 * PANEL_PAD

# Between-round pause (frames at current FPS)
ROUND_PAUSE_MS = 2500   # ms to show the winner screen before next round


def load_agents(device, use_best: bool):
    
    obs_size  = SnakeEnv.OBS_SIZE
    n_actions = SnakeEnv.N_ACTIONS

    if use_best:
        model_dir = MODELS_BEST_DIR
        filename  = lambda key: f"{key}_best.pt"
    else:
        model_dir = MODELS_DIR
        filename  = lambda key: f"{key}.pt"

    configs = [
        ("DQN",     "dqn",     DQNAgent(obs_size,   n_actions, device)),
        ("DDQN",    "ddqn",    DDQNAgent(obs_size,  n_actions, device)),
        ("Rainbow", "rainbow", RainbowAgent(obs_size, n_actions, device)),
    ]

    loaded = []
    for label, key, agent in configs:
        path = os.path.join(model_dir, filename(key))
        if os.path.exists(path):
            agent.load(path)
            agent.online.eval()
            print(f"[OK]  Loaded {label} from {path}")
        else:
            print(f"[--]  No model found for {label} at {path} — using untrained agent")
        loaded.append((label, agent))

    return loaded


def get_greedy_action(agent, state):
    
    with torch.no_grad():
        s = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        if agent.noisy:
            # Keep noise OFF for greedy play (uses mean parameters as per NoisyNet paper)
            agent.online.eval()
        return int(agent.online(s).argmax().item())


def draw_panel(surface, state, label, agent_color, font, small_font,
               dead=False, winner=False):
   
    surface.fill(PANEL_BG)

    # Grid lines
    for i in range(GRID_SIZE + 1):
        pygame.draw.line(surface, GRID_LINE, (i * CELL, 0), (i * CELL, PANEL_H))
        pygame.draw.line(surface, GRID_LINE, (0, i * CELL), (PANEL_W, i * CELL))

    body = state["body"]
    food = state["food"]

    # Food
    fr, fc = food
    pygame.draw.ellipse(surface, FOOD_COLOR,
                        pygame.Rect(fc * CELL + 2, fr * CELL + 2, CELL - 4, CELL - 4))

    # Body
    for seg_idx in range(len(body) - 1, 0, -1):
        r, c = body[seg_idx]
        fade = max(60, 200 - seg_idx * 6)
        color = (
            min(255, int(SNAKE_BODY[0] * fade // 200)),
            min(255, int(SNAKE_BODY[1] * fade // 200)),
            min(255, int(SNAKE_BODY[2] * fade // 200)),
        )
        pygame.draw.rect(surface, color,
                         pygame.Rect(c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2),
                         border_radius=4)

    # Head
    hr, hc = body[0]
    head_color = DEAD_COLOR if dead else SNAKE_HEAD
    pygame.draw.rect(surface, head_color,
                     pygame.Rect(hc * CELL + 1, hr * CELL + 1, CELL - 2, CELL - 2),
                     border_radius=6)
    if not dead:
        d = state["direction"]
        offsets = {0: [(5, 4), (5, 12)], 1: [(4, 14), (12, 14)],
                   2: [(12, 4), (12, 12)], 3: [(4, 4), (12, 4)]}
        for ey, ex in offsets.get(d, [(5, 5), (5, 13)]):
            pygame.draw.circle(surface, (10, 10, 10),
                               (hc * CELL + ex, hr * CELL + ey), 3)

    # Label
    lbl = font.render(label, True, agent_color)
    surface.blit(lbl, (PANEL_W // 2 - lbl.get_width() // 2, 4))

    # Score
    score_txt = small_font.render(f"Score: {state['score']}", True, SCORE_GLOW)
    surface.blit(score_txt, (4, PANEL_H - score_txt.get_height() - 4))

    # Border — gold if winner
    border_color = WIN_FLASH if winner else (DEAD_COLOR if dead else agent_color)
    border_w     = 4 if winner else 2
    pygame.draw.rect(surface, border_color, (0, 0, PANEL_W, PANEL_H), border_w, border_radius=6)

    # Dark overlay for dead panels
    if dead and not winner:
        overlay = pygame.Surface((PANEL_W, PANEL_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        surface.blit(overlay, (0, 0))

    # "WINNER" badge
    if winner:
        win_font = pygame.font.SysFont("Consolas", 28, bold=True)
        w_txt = win_font.render("WINNER", True, WIN_FLASH)
        surface.blit(w_txt, (PANEL_W // 2 - w_txt.get_width() // 2,
                              PANEL_H // 2 - w_txt.get_height() // 2))


def announce_round(round_num, scores, labels):
    """Print round results to terminal."""
    winner_idx  = int(np.argmax(scores))
    winner_name = labels[winner_idx]
    print(f"\n{'─'*45}")
    print(f"  Tur {round_num} bitti!")
    for i, (lbl, sc) in enumerate(zip(labels, scores)):
        marker = " ← KAZANAN" if i == winner_idx else ""
        print(f"    {lbl:<10} skor: {sc}{marker}")
    print(f"  Kazanan: {winner_name}")
    print(f"{'─'*45}")
    return winner_idx


def main():
    parser = argparse.ArgumentParser(description="Snake RL Player")
    parser.add_argument("--best", action="store_true",
                        help="Load best models from models_best/ instead of models/")
    args = parser.parse_args()

    pygame.init()
    pygame.display.set_caption("Snake RL — DQN | DDQN | Rainbow")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()

    font       = pygame.font.SysFont("Consolas", 18, bold=True)
    small_font = pygame.font.SysFont("Consolas", 14)
    big_font   = pygame.font.SysFont("Consolas", 22, bold=True)
    huge_font  = pygame.font.SysFont("Consolas", 26, bold=True)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    src      = "best models" if args.best else "final models"
    print(f"[Device] {('cuda' if torch.cuda.is_available() else 'cpu')} ({gpu_name})")
    print(f"[Mode]   Loading {src}")

    agents = load_agents(device, use_best=args.best)
    labels = [lbl for lbl, _ in agents]
    n      = len(agents)

    panels = [pygame.Surface((PANEL_W, PANEL_H)) for _ in agents]

    fps        = FPS_DEFAULT
    paused     = False
    running    = True

    # ── Round state ───────────────────────────────────────────────────
    round_num     = 1
    round_scores  = [0] * n   # score for THIS round
    cumul_wins    = [0] * n   # total wins across all rounds
    cumul_scores  = [0] * n   # cumulative score for avg

    envs   = [SnakeEnv(grid_size=GRID_SIZE) for _ in agents]
    states = [env.reset() for env in envs]
    deads  = [False] * n
    episode_scores = [0] * n  # score of the current episode (reset each death)

    # Between-round display
    between_round      = False
    between_round_timer = 0
    winner_idx         = -1

    print(f"\n[Tur {round_num} başlıyor]  Q=hızlandır  E=yavaşlat  SPACE=duraklat  ESC=çıkış\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_q:
                    fps = min(FPS_MAX, fps + FPS_STEP)
                    print(f"[Hız] FPS: {fps}")
                if event.key == pygame.K_e:
                    fps = max(FPS_MIN, fps - FPS_STEP)
                    print(f"[Hız] FPS: {fps}")

        # ── Between-round pause ───────────────────────────────────────
        if between_round:
            between_round_timer -= clock.get_time()
            if between_round_timer <= 0:
                # Start next round
                between_round  = False
                round_num     += 1
                round_scores   = [0] * n
                deads          = [False] * n
                episode_scores = [0] * n
                states         = [env.reset() for env in envs]
                winner_idx     = -1
                print(f"\n[Tur {round_num} başlıyor]\n")

        # ── Game step ─────────────────────────────────────────────────
        elif not paused:
            for i, (label, agent) in enumerate(agents):
                if deads[i]:
                    continue  # stay dead until round resets
                action = get_greedy_action(agent, states[i])
                next_state, reward, done, info = envs[i].step(action)
                states[i] = next_state
                if done:
                    episode_scores[i] = info["score"]
                    round_scores[i]   = info["score"]
                    deads[i]          = True

            # Check if round is over (all dead)
            if all(deads):
                winner_idx = announce_round(round_num, round_scores, labels)
                cumul_wins[winner_idx]  += 1
                for i in range(n):
                    cumul_scores[i] += round_scores[i]
                between_round       = True
                between_round_timer = ROUND_PAUSE_MS

        # ── Draw ──────────────────────────────────────────────────────
        screen.fill(BG)

        # Title
        mode_str = "  [BEST MODELS]" if args.best else "  [FINAL MODELS]"
        title = big_font.render(
            f"Snake RL — DQN | DDQN | Rainbow{mode_str}", True, TITLE_COLOR)
        screen.blit(title, (WIN_W // 2 - title.get_width() // 2, 10))

        # Controls hint
        ctrl_txt = small_font.render(
            f"[Q] hızlandır  [E] yavaşlat  [SPACE] duraklat  [ESC] çıkış  FPS:{fps}",
            True, (80, 90, 120))
        screen.blit(ctrl_txt, (WIN_W // 2 - ctrl_txt.get_width() // 2, 36))

        # Round info
        round_txt = font.render(f"Tur {round_num}", True, TEXT_COLOR)
        screen.blit(round_txt, (WIN_W // 2 - round_txt.get_width() // 2, 56))

        # Panels
        for i, (label, agent) in enumerate(agents):
            st      = envs[i].get_render_state()
            ac      = AGENT_COLORS.get(label, TEXT_COLOR)
            is_win  = between_round and (i == winner_idx)
            draw_panel(panels[i], st, label, ac, font, small_font,
                       dead=deads[i], winner=is_win)
            x = PANEL_PAD + i * (PANEL_W + PANEL_PAD)
            screen.blit(panels[i], (x, INFO_HEIGHT))

        # Footer stats
        footer_y = INFO_HEIGHT + PANEL_H + PANEL_PAD
        for i, (label, _) in enumerate(agents):
            ac  = AGENT_COLORS.get(label, TEXT_COLOR)
            avg = cumul_scores[i] / max(1, round_num - 1) if round_num > 1 else 0
            info_line = (f"{label}: bu_tur={round_scores[i]}  "
                         f"kazanma={cumul_wins[i]}  avg={avg:.1f}")
            txt = small_font.render(info_line, True, ac)
            x   = PANEL_PAD + i * (PANEL_W + PANEL_PAD)
            screen.blit(txt, (x, footer_y))

        # Between-round winner overlay
        if between_round and winner_idx >= 0:
            w_label = labels[winner_idx]
            w_color = AGENT_COLORS.get(w_label, WIN_FLASH)
            overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 60))
            screen.blit(overlay, (0, 0))
            msg1 = huge_font.render(f"Tur {round_num} bitti!", True, WIN_FLASH)
            msg2 = big_font.render(f"Kazanan: {w_label}  (skor: {round_scores[winner_idx]})",
                                   True, w_color)
            msg3 = small_font.render("Sonraki tura geçiliyor...", True, TEXT_COLOR)
            cy = WIN_H // 2
            screen.blit(msg1, (WIN_W // 2 - msg1.get_width() // 2, cy - 56))
            screen.blit(msg2, (WIN_W // 2 - msg2.get_width() // 2, cy - 10))
            screen.blit(msg3, (WIN_W // 2 - msg3.get_width() // 2, cy + 34))

        # Pause overlay
        if paused:
            ov = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 120))
            screen.blit(ov, (0, 0))
            p_txt = big_font.render("DURAKLATILDI — SPACE ile devam et", True, (255, 220, 80))
            screen.blit(p_txt, (WIN_W // 2 - p_txt.get_width() // 2,
                                 WIN_H // 2 - p_txt.get_height() // 2))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
