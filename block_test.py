import numpy as np
from random import random
import curses
import time
import model

GRID_SIZE = 32
BLOCK_SIZE = 4
BLOCK_SIZE_HALF = int(round(BLOCK_SIZE / 2))
BLOCK_MASS = 0.1
FRICTION = 0.01
TIMESTEP = 1e-3 


class block_test():
    def __init__(self, use_curses=True):
        self._grid = np.zeros([GRID_SIZE, GRID_SIZE], dtype=np.int64)
        self.reset()
        if curses:
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self._render()

    def reset(self):
        self._grid[:, :] = 0
        self.x = (BLOCK_SIZE / 2) + (random() * (GRID_SIZE - BLOCK_SIZE))
        self.y = (BLOCK_SIZE / 2) + (random() * (GRID_SIZE - BLOCK_SIZE))
        self.pixel_x = 0
        self.pixel_y = 0
        self.pixel_x_last = 0
        self.pixel_y_last = 0
        self.vx = 1000. * random()
        self.vy = 1000. * random()

    @property
    def grid(self):
        return self._grid

    def _render(self):
        """
        Render block on grid
        """
        self.pixel_x_last = self.pixel_x
        self.pixel_y_last = self.pixel_y
        pixel_x_corner = int(round(self.x - BLOCK_SIZE_HALF))
        pixel_y_corner = int(round(self.y - BLOCK_SIZE_HALF))
        # Check bounds
        pixel_x_bounded = 0 if pixel_x_corner < 0 else pixel_x_corner
        pixel_y_bounded = 0 if pixel_y_corner < 0 else pixel_y_corner
        self.pixel_x = ((GRID_SIZE - BLOCK_SIZE) if pixel_x_bounded >
                   (GRID_SIZE - BLOCK_SIZE) else pixel_x_bounded)
        self.pixel_y = ((GRID_SIZE - BLOCK_SIZE) if pixel_y_bounded >
                   (GRID_SIZE - BLOCK_SIZE) else pixel_y_bounded)
        # Zeroise last position
        self._grid[self.pixel_x_last: self.pixel_x_last + BLOCK_SIZE,
                  self.pixel_y_last: self.pixel_y_last + BLOCK_SIZE] = 0
        # Add block
        self._grid[self.pixel_x: self.pixel_x + BLOCK_SIZE,
                  self.pixel_y: self.pixel_y + BLOCK_SIZE] = 1

    def show(self, grid):
        """
        Print out the grid
        """
        for i in range(GRID_SIZE):
            grid_line = ""
            for j in range(GRID_SIZE):
                grid_line += str(grid[i, j]) + " "
            self.stdscr.addstr(i, 0, grid_line)
        self.stdscr.refresh()

    def step(self, fx, fy):
        """
        Step physics ENGINE!
        """
        self.vx += ((fx * TIMESTEP) / BLOCK_MASS) - self.vx * FRICTION
        # Bounce off wall
        if ((self.x < BLOCK_SIZE_HALF) or self.x >
                (GRID_SIZE - BLOCK_SIZE_HALF)):
            self.vx = - self.vx
        self.x += self.vx * TIMESTEP
        self.vy += ((fy * TIMESTEP) / BLOCK_MASS) - self.vy * FRICTION
        # Bounce off wall
        if ((self.y < BLOCK_SIZE_HALF) or self.y >
                (GRID_SIZE - BLOCK_SIZE_HALF)):
            self.vy = - self.vy
        self.y += self.vy * TIMESTEP
        self._render()
        return self._grid


try:
    a = block_test(use_curses=True)
    my_model = model.Model()
    # State is 4 Grids to satisfy markov property
    s0 = np.zeros([1, 1, GRID_SIZE, 4 * GRID_SIZE], dtype=np.float32)
    s1 = np.zeros([1, 1, GRID_SIZE, 4 * GRID_SIZE], dtype=np.float32)
    # Run first four steps to get initial observation
    for k in range(1000000000):
        a.reset()
        for i in range(4):
            observation = a.step(0., 0.)
            s0[:, :, :, (i * GRID_SIZE): ((i * GRID_SIZE) + GRID_SIZE)] = observation
        for i in range(4):
            observation = a.step(0., 0.)
            s1[:, :, :, (i * GRID_SIZE): ((i * GRID_SIZE) + GRID_SIZE)] = observation
        #print("diff: {}".format(np.array_equal(s0, s1)))
        y1 = my_model.train(s0, s1)
        if k % 100 == 0:
            # print(s0.shape)
            for i in range(4):
                time.sleep(0.1)
                s0_frame = s0[:, :, :, (i * GRID_SIZE): (i * GRID_SIZE + GRID_SIZE)]
                a.show(np.rint(s0_frame.reshape([GRID_SIZE, GRID_SIZE])).astype(np.int64))
            for i in range(4):
                time.sleep(0.1)
                s1_frame = s1[:, :, :, (i * GRID_SIZE): (i * GRID_SIZE + GRID_SIZE)]
                a.show(np.rint(s1_frame.reshape([GRID_SIZE, GRID_SIZE])).astype(np.int64))
            for i in range(4):
                time.sleep(0.1)
                s0_frame = s0[:, :, :, (i * GRID_SIZE): (i * GRID_SIZE + GRID_SIZE)]
                a.show(np.rint(s0_frame.reshape([GRID_SIZE, GRID_SIZE])).astype(np.int64))
            for i in range(4):
                time.sleep(0.1)
                y1_frame = y1[:, :, :, (i * GRID_SIZE): (i * GRID_SIZE + GRID_SIZE)]
                a.show(np.rint(y1_frame.reshape([GRID_SIZE, GRID_SIZE])).astype(np.int64))
    """
    for i in range(1000):
        my_model.train(s0, s1)
        s1 = s0
        for j in range(4):
            time.sleep(0.01)
            observation = a.step(0 * (random() - 0.5), 0 * (random() - 0.5))
            s0[:, :, :, (j * GRID_SIZE): ((j * GRID_SIZE) + GRID_SIZE)] = observation
        #a.show(a.grid)
    x = s
    """
    #a.show(np.rint(out[:, 0:GRID_SIZE]).astype(np.int64))
finally:
    curses.echo()
    curses.nocbreak()
    curses.endwin()
