import os
from datetime import datetime
from random import random

import numpy as np
from PIL import Image

GRID_SIZE = 32
BLOCK_SIZE = 4
BLOCK_SIZE_HALF = int(round(BLOCK_SIZE / 2))
BLOCK_MASS = 0.1
FRICTION = 0.01
TIMESTEP = 1e-3
FORCE_SCALE = 100000.
BATCH_SIZE = 64
DEFAULT_RENDER_PORT = 8123
FRAME_DIR = "frames"
# Image depth is frame at each timestep.
IMAGE_DEPTH = 1
FRAMES = 4


class BlockSys:
    def __init__(self):
        self._grid = np.zeros([GRID_SIZE, GRID_SIZE], dtype=np.float32)
        self.reset()

    def reset(self):
        self._grid[:, :] = 0.
        self.x = (BLOCK_SIZE / 2) + (random() * (GRID_SIZE - BLOCK_SIZE))
        self.y = (BLOCK_SIZE / 2) + (random() * (GRID_SIZE - BLOCK_SIZE))
        self.pixel_x = 0
        self.pixel_y = 0
        self.pixel_x_last = 0
        self.pixel_y_last = 0
        # self.vx = 0.
        # self.vy = 0.
        self.vx = 1000. * random()
        self.vy = 1000. * random()

    @property
    def grid(self):
        return self._grid

    def _rasterize(self):
        """
        Rasterize block on grid
        """
        self.pixel_x_last = self.pixel_x
        self.pixel_y_last = self.pixel_y
        pixel_x_corner = int(round(self.x - BLOCK_SIZE_HALF))
        pixel_y_corner = int(round(self.y - BLOCK_SIZE_HALF))
        # Check bounds
        pixel_x_bounded = 0 if pixel_x_corner < 0 else pixel_x_corner
        pixel_y_bounded = 0 if pixel_y_corner < 0 else pixel_y_corner
        self.pixel_x = (
            (GRID_SIZE - BLOCK_SIZE)
            if pixel_x_bounded > (GRID_SIZE - BLOCK_SIZE)
            else pixel_x_bounded
        )
        self.pixel_y = (
            (GRID_SIZE - BLOCK_SIZE)
            if pixel_y_bounded > (GRID_SIZE - BLOCK_SIZE)
            else pixel_y_bounded
        )
        # Zeroise last position
        self._grid[
            self.pixel_x_last : self.pixel_x_last + BLOCK_SIZE,
            self.pixel_y_last : self.pixel_y_last + BLOCK_SIZE,
        ] = 0.
        # Add block
        self._grid[
            self.pixel_x : self.pixel_x + BLOCK_SIZE,
            self.pixel_y : self.pixel_y + BLOCK_SIZE,
        ] = 1.

    def step(self, fx, fy):
        """
        Step physics ENGINE!
        """
        self.vx += ((fx * TIMESTEP) / BLOCK_MASS) - self.vx * FRICTION
        # Bounce off wall
        if self.x < BLOCK_SIZE_HALF:
            self.x = BLOCK_SIZE_HALF
            self.vx = -self.vx
        if self.x > (GRID_SIZE - BLOCK_SIZE_HALF):
            self.x = GRID_SIZE - BLOCK_SIZE_HALF
            self.vx = -self.vx
        self.x += self.vx * TIMESTEP
        self.vy += ((fy * TIMESTEP) / BLOCK_MASS) - self.vy * FRICTION
        # Bounce off wall
        if self.y < BLOCK_SIZE_HALF:
            self.v = BLOCK_SIZE_HALF
            self.vy = -self.vy
        if self.y > (GRID_SIZE - BLOCK_SIZE_HALF):
            self.v = GRID_SIZE - BLOCK_SIZE_HALF
            self.vy = -self.vy
        self.y += self.vy * TIMESTEP
        self._rasterize()
        return self._grid


"""
if __name__ == "__main__":
    try:
        a = BlockSys(use_curses=True)
        my_model = model.Model()
        # State is 4 Grids to satisfy markov property
        s0 = np.zeros([BATCH_SIZE, 1, GRID_SIZE, 4 * GRID_SIZE], dtype=np.float32)
        s1 = np.zeros([BATCH_SIZE, 1, GRID_SIZE, 4 * GRID_SIZE],
                      dtype=np.float32)
        force = np.zeros([BATCH_SIZE, 2], dtype=np.float32)
        # Run first four steps to get initial observation
        batch_idx = 0
        batch_count = 0
        for k in range(1000000000):
            a.reset()
            for i in range(4):
                observation = a.step(0., 0.)
                s0[batch_idx, :, :,
                   (i * GRID_SIZE): ((i * GRID_SIZE) + GRID_SIZE)] = obsv
            force[batch_idx, :] = np.tile(np.array([FORCE_SCALE * (random() - 0.5),
                                            FORCE_SCALE * (random() - 0.5)],
                                            dtype=np.float32), 1)
            for i in range(4):
                obsv = a.step(force[batch_idx, 0], force[batch_idx, 1])
                s1[batch_idx, :, :, (i * GRID_SIZE): ((i * GRID_SIZE) + GRID_SIZE)] = obsv
            # print("diff: {}".format(np.array_equal(s0, s1)))
            if batch_idx < 31:
                batch_idx += 1
            else:
                # Train with normalized inputs
                y1 = my_model.train(s0 - 0.5, s1, force / FORCE_SCALE)
                batch_idx = 0
                batch_count += 1
                if batch_count % 30 == 0:
                    # print(s0.shape)
                    for i in range(4):
                        #time.sleep(0.1)
                        s0_frame = s0[0, :, :, (i * GRID_SIZE): (i * GRID_SIZE + GRID_SIZE)]
                        #a.show(np.rint(s0_frame).astype(np.int64))
                    for i in range(4):
                        time.sleep(0.1)
                        s1_frame = s1[0, :, :, (i * GRID_SIZE): (i * GRID_SIZE + GRID_SIZE)]
                        a.show(2 * np.rint(s1_frame).astype(np.int64))
                    for i in range(4):
                        #time.sleep(0.1)
                        s0_frame = s0[0, :, :, (i * GRID_SIZE): (i * GRID_SIZE + GRID_SIZE)]
                        #a.show(3 * np.rint(s0_frame).astype(np.int64))
                    for i in range(4):
                        time.sleep(0.1)
                        y1_frame = y1[0, :, :, (i * GRID_SIZE): (i * GRID_SIZE + GRID_SIZE)]
                        a.show(4 * np.rint(y1_frame).astype(np.int64))
"""


def render(grid, suffix=""):
    grid255 = np.clip(255. * grid, 0., 255.)
    grid_uint = np.rint(grid255).astype("uint8")
    im = Image.fromarray(grid_uint, mode="RGB").resize((640, 640))
    im.save(os.path.join(FRAME_DIR, "{}{}.jpeg".format(str(datetime.now()), suffix)))
    return im
