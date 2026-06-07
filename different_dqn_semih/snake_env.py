import numpy as np
from collections import deque

# Directions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
DIR_VEC = {
    0: (-1,  0),
    1: ( 0,  1),
    2: ( 1,  0),
    3: ( 0, -1),
}


TURN = {
    0: {0: 0, 1: 1, 2: 2, 3: 3},  # düz
    1: {0: 1, 1: 2, 2: 3, 3: 0},  # sağ
    2: {0: 3, 1: 0, 2: 1, 3: 2},  # sol
}

# raycast
RAY_OFFSETS = [0, 1, 2, 3, 4, 5, 6, 7]  # in 45° units


class SnakeEnv:


    OBS_SIZE = 32
    N_ACTIONS = 3

    def __init__(self, grid_size=20, max_steps=None):
        self.grid_size = grid_size
        self.max_steps = max_steps or grid_size * grid_size * 4
        self.n_rays = 8
        self._reset_state()

    def reset(self):
        self._reset_state()
        return self._get_obs()

    def step(self, action):

        self.steps += 1

        # Update direction
        self.direction = TURN[action][self.direction]
        dr, dc = DIR_VEC[self.direction]


        hr, hc = self.head
        new_head = (hr + dr, hc + dc)

 
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            return self._get_obs(), -10.0, True, {"score": self.score}


        if new_head in self.body_set:
            return self._get_obs(), -10.0, True, {"score": self.score}


        old_dist = self._manhattan(self.head, self.food)

        self.body.appendleft(new_head)
        self.body_set.add(new_head)
        self.head = new_head

        ate_food = (new_head == self.food)
        reward = 0.0

        if ate_food:
            self.score += 1
            reward = 10.0
            self._place_food()
        else:
            # (O(1) deque pop)
            tail = self.body.pop()
            self.body_set.discard(tail)


            new_dist = self._manhattan(self.head, self.food)
            reward = 1.0 if new_dist < old_dist else -1.5

        # Step limit
        done = self.steps >= self.max_steps
        if done and reward >= 0:
            reward = -5.0  # mild penalty for timing out

        return self._get_obs(), reward, done, {"score": self.score}


    def _get_obs(self):
        rays = self._cast_rays()                   # 24 values
        dir_oh = np.zeros(4, dtype=np.float32)
        dir_oh[self.direction] = 1.0              # 4 values
        food_dir = self._food_direction()         # 4 values
        return np.concatenate([rays, dir_oh, food_dir])

    def _cast_rays(self):
        """
        Cast 8 rays in 45° increments relative to current facing.
        Returns flat array of shape (24,): [wall, body, food] per ray.
        Distances normalized to [0,1].
        """
        result = np.zeros(self.n_rays * 3, dtype=np.float32)
        hr, hc = self.head
        max_d = float(self.grid_size)

        # 45-degree direction vectors (row, col)
        ALL_DIRS = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]

        # Current direction index in ALL_DIRS (matches DIR_VEC)
        # DIR_VEC: 0=(-1,0)=UP, 1=(0,1)=RIGHT, 2=(1,0)=DOWN, 3=(0,-1)=LEFT
        # Map to 45-deg index: UP=0, RIGHT=2, DOWN=4, LEFT=6
        dir_to_idx = {0: 0, 1: 2, 2: 4, 3: 6}
        base = dir_to_idx[self.direction]

        for i, offset in enumerate(RAY_OFFSETS):
            rd, cd = ALL_DIRS[(base + offset) % 8]
            body_d = 0.0   # 0 means not found (far)
            food_d = 0.0

            r, c = hr + rd, hc + cd
            dist = 1.0
            found_body = False
            found_food = False
            while 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                pos = (r, c)
                if not found_body and pos in self.body_set:
                    body_d = 1.0 - min(dist / max_d, 1.0)  # closer → higher
                    found_body = True
                if not found_food and pos == self.food:
                    food_d = 1.0 - min(dist / max_d, 1.0)
                    found_food = True
                r += rd
                c += cd
                dist += 1.0

            # Wall: how close is the wall in this direction (1=right next to it, 0=far)
            wall_d = 1.0 - min((dist - 1.0) / max_d, 1.0)

            idx = i * 3
            result[idx]     = wall_d
            result[idx + 1] = body_d
            result[idx + 2] = food_d

        return result

    def _food_direction(self):
        """4-value one-hot-ish: [food_up, food_right, food_down, food_left]."""
        fr, fc = self.food
        hr, hc = self.head
        return np.array([
            1.0 if fr < hr else 0.0,
            1.0 if fc > hc else 0.0,
            1.0 if fr > hr else 0.0,
            1.0 if fc < hc else 0.0,
        ], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _reset_state(self):
        g = self.grid_size
        mid = g // 2
        # Start with length 3, moving right
        self.head = (mid, mid)
        self.body = deque([(mid, mid), (mid, mid - 1), (mid, mid - 2)])
        self.body_set = set(self.body)
        self.direction = 1  # RIGHT
        self.score = 0
        self.steps = 0
        self._place_food()

    def _place_food(self):
        """Place food at a random free cell. O(free_cells) worst case."""
        g = self.grid_size
        while True:
            r = np.random.randint(0, g)
            c = np.random.randint(0, g)
            pos = (r, c)
            if pos not in self.body_set:
                self.food = pos
                break

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ------------------------------------------------------------------ #
    # Rendering helpers (used by play.py)                                  #
    # ------------------------------------------------------------------ #

    def get_render_state(self):
        """Return state dict for external rendering."""
        return {
            "body":      list(self.body),
            "head":      self.head,
            "food":      self.food,
            "score":     self.score,
            "direction": self.direction,
            "grid_size": self.grid_size,
        }
