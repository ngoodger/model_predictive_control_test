import pygame, random
import numpy as np
from block_sys import FORCE_SCALE, FRAMES, IMAGE_DEPTH, GRID_SIZE
import block_sys as bs
import torch
import time

PIXELS = 640
LEFT = 1
RIGHT = 3

PIXELS_PER_BLOCK = int(PIXELS / GRID_SIZE)
print(PIXELS_PER_BLOCK)

screen = pygame.display.set_mode((PIXELS, PIXELS))

screen_draw = pygame.Surface((PIXELS, PIXELS))

draw_on = False
last_pos = (0, 0)
color = (255, 128, 0)
radius = 10
device = "cpu"
SCALE_ARR = np.ones((PIXELS_PER_BLOCK, PIXELS_PER_BLOCK, 1))

my_block_sys = bs.BlockSys()
my_block_sys_target = bs.BlockSys()
model = torch.load("recurrent_model.pt", map_location=device)
policy = torch.load("my_policy.pt", map_location=device)
model_input_cnn = torch.load("input_cnn.pt", map_location=device)
input_cnn = torch.load("input_cnn.pt", map_location=device)
force_0 = np.zeros([1, 2], dtype=np.float32)
s0 = np.zeros([1, IMAGE_DEPTH, GRID_SIZE, GRID_SIZE, FRAMES], dtype=np.float32)
s1_target = np.zeros([1, IMAGE_DEPTH, GRID_SIZE, GRID_SIZE, 4], dtype=np.float32)
force_0[0, :] = np.array(
    [random.random() - 0.5, random.random() - 0.5], dtype=np.float32
).reshape([1, 2])

try:
    while True:
        # time.sleep(0.05)
        e = pygame.event.poll()
        if e.type == pygame.QUIT:
            raise StopIteration
        if e.type == pygame.MOUSEBUTTONDOWN:
            leftclick, middleclick, rightclick = pygame.mouse.get_pressed()
            if leftclick:
                color = (255, 255, 255)
            else:
                color = (0, 0, 0)
            x, y = e.pos
            x_block = int(x / PIXELS_PER_BLOCK)
            y_block = int(y / PIXELS_PER_BLOCK)
            pygame.draw.rect(
                screen_draw,
                color,
                (
                    x_block * PIXELS_PER_BLOCK,
                    y_block * PIXELS_PER_BLOCK,
                    PIXELS_PER_BLOCK,
                    PIXELS_PER_BLOCK,
                ),
            )
            draw_on = True
        if e.type == pygame.MOUSEBUTTONUP:
            draw_on = False
        if e.type == pygame.MOUSEMOTION:
            leftclick, middleclick, rightclick = pygame.mouse.get_pressed()
            if leftclick:
                color = (255, 255, 255)
            else:
                color = (0, 0, 0)
            if draw_on:
                x, y = e.pos
                x_block = int(x / PIXELS_PER_BLOCK)
                y_block = int(y / PIXELS_PER_BLOCK)
                pygame.draw.rect(
                    screen_draw,
                    color,
                    (
                        x_block * PIXELS_PER_BLOCK,
                        y_block * PIXELS_PER_BLOCK,
                        PIXELS_PER_BLOCK,
                        PIXELS_PER_BLOCK,
                    ),
                )
                # roundline(screen, color, e.pos, last_pos,  radius)
            last_pos = e.pos
        pygame.display.flip()
        # my_surface = pygame..get_surface()
        a = pygame.surfarray.array3d(screen_draw)

        i = 0
        s0[0, 0, :, :, i] = my_block_sys.step(
            FORCE_SCALE * (force_0[0, 0]), FORCE_SCALE * (force_0[0, 1])
        )
        test = np.zeros((GRID_SIZE, GRID_SIZE, 3))
        test[:, :, 0] = np.rint(255 * s0[0, 0, :, :, i])
        test[:, :, 1] = np.rint(255 * s0[0, 0, :, :, i])
        test[:, :, 2] = np.rint(255 * s0[0, 0, :, :, i])
        block_img = np.kron(test, SCALE_ARR) + a

        new_surf = pygame.pixelcopy.make_surface(block_img.astype(np.uint8))
        screen.blit(new_surf, (0, 0))


except StopIteration:
    pass

pygame.quit()
