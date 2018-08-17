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
    def __init__(self):
        self._grid = np.zeros([GRID_SIZE, GRID_SIZE], dtype=np.int64)
        self.x = (BLOCK_SIZE / 2) + (random() * (GRID_SIZE - BLOCK_SIZE))
        self.y = (BLOCK_SIZE / 2) + (random() * (GRID_SIZE - BLOCK_SIZE))
        self.pixel_x = 0
        self.pixel_y = 0
        self.pixel_x_last = 0
        self.pixel_y_last = 0
        self.vx = 1000. * random()
        self.vy = 1000. * random()
        self._test_grid = "hello"
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self._render()

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
    cnn_model = model.ConvNet()
    print(cnn_model)
    a = block_test()
    # State is 4 Grids to satisfy markov property
    s = np.zeros([1, 1, GRID_SIZE, 4 * GRID_SIZE], dtype=np.float32)
    # Run first four steps to get initial observation
    for i in range(4):
        observation = a.step(0., 0.)
        s[:, :, :, (i * GRID_SIZE): ((i * GRID_SIZE) + GRID_SIZE)] = observation
    x = s
    out = cnn_model.forward(x)
    print(out.shape)
    a.show(out)
    for i in range(100):
        time.sleep(0.1)
        #observation = a.step(0., 0.)
        a.show(a.grid)
finally:
    curses.echo()
    curses.nocbreak()
    curses.endwin()
