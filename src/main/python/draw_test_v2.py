import pygame
import random
import numpy as np
from block_sys import FORCE_SCALE, FRAMES, IMAGE_DEPTH, GRID_SIZE
import block_sys as bs
import torch

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
block_screen = np.zeros((GRID_SIZE, GRID_SIZE, 3))

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

target_temp = np.zeros((1, 1, GRID_SIZE, GRID_SIZE, FRAMES), dtype=np.float32)
model_temp = np.zeros((1, 1, GRID_SIZE, GRID_SIZE, FRAMES), dtype=np.float32)
model_temp_last = np.zeros((1, 1, GRID_SIZE, GRID_SIZE, FRAMES), dtype=np.float32)
model_screen = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
frame = 0
freeze = False

print("")
print("-------------What am I seeing?-------------")
print("Green block: Block location in simulated world")
print("Red block / cloud: Model predicted location")
print("Drawn Blue square: Target image drawing")
print("")
print("-------------What's happening?-------------")
print(
    "Policy is running that has learnt to make the current image look\n"
    " like the target image"
)
print(
    "Model prediction 4 frames ahead of block is shown in red however\n"
    " it was only used for training not here and is just shown for fun."
)

print("")
print("-------------Controls-------------")
print("Space bar: Toggle start stop physics")
print("Left mouse button: Draw target image")
print("Right mouse button: Erase target image")
print("")

try:
    while True:
        # time.sleep(0.05)
        e = pygame.event.poll()
        if e.type == pygame.QUIT:
            raise StopIteration
        if e.type == pygame.MOUSEBUTTONDOWN:
            leftclick, middleclick, rightclick = pygame.mouse.get_pressed()
            if leftclick:
                color = (0, 255, 0)
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
        if e.type == pygame.KEYDOWN:
            if pygame.K_SPACE:
                if freeze:
                    freeze = False
                else:
                    freeze = True
        if e.type == pygame.MOUSEBUTTONUP:
            draw_on = False
        if e.type == pygame.MOUSEMOTION:
            leftclick, middleclick, rightclick = pygame.mouse.get_pressed()
            if leftclick:
                color = (0, 255, 0)
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
        target_screen = pygame.surfarray.array3d(screen_draw)

        if not freeze:
            model_temp_last = model_temp
            s0[0, 0, :, :, frame] = my_block_sys.step(
                FORCE_SCALE * (force_0[0, 0]), FORCE_SCALE * (force_0[0, 1])
            )
            if frame < FRAMES - 1:
                frame += 1
            else:
                frame = 0
                force_0_tensor = torch.from_numpy(force_0).to(device)
                start = torch.from_numpy(s0).to(device)
                target_temp[0, 0, :, :, 0] = (
                    target_screen[1::PIXELS_PER_BLOCK, 1::PIXELS_PER_BLOCK, 1] / 255
                )
                target_temp[0, 0, :, :, 1] = target_temp[0, 0, :, :, 0]
                target_temp[0, 0, :, :, 2] = target_temp[0, 0, :, :, 0]
                target_temp[0, 0, :, :, 3] = target_temp[0, 0, :, :, 0]
                s1_target = np.rint(target_temp)
                target = torch.from_numpy(s1_target).to(device)
                out_target_cnn_flat = input_cnn.forward(target)
                if frame == 0:
                    out_start_cnn_flat_model = model_input_cnn.forward(start)
                    out_start_cnn_flat = input_cnn.forward(start)
                    force_1_tensor, out_target_cnn_layer = policy.forward(
                        force_0_tensor,
                        out_start_cnn_flat,
                        out_target_cnn_flat,
                        None,
                        first_iteration=True,
                    )
                    logits, out, recurrent_state = model.forward(
                        out_start_cnn_flat_model,
                        None,
                        force_0_tensor,
                        force_1_tensor,
                        first_iteration=True,
                    )
                else:
                    out_start_cnn_flat_model = model_input_cnn.forward(start)
                    out_start_cnn_flat = input_cnn.forward(start)
                    force_1_tensor, _ = policy.forward(
                        force_0_tensor,
                        out_start_cnn_flat,
                        out_target_cnn_flat,
                        out_target_cnn_layer=out_target_cnn_layer,
                        first_iteration=False,
                    )
                    # DELETE THIS
                    # force_1_tensor = force_0_tensor
                    logits, out, recurrent_state = model.forward(
                        out_start_cnn_flat_model,
                        recurrent_state,
                        force_0_tensor,
                        force_1_tensor,
                        first_iteration=False,
                    )
                force_1 = force_1_tensor.data.numpy()
                force_0 = force_1
                out_numpy = out.data.numpy()
                model_temp[0, 0, :, :, 0] = out_numpy[0, 0, :, :, 0] * 255
                model_temp[0, 0, :, :, 1] = out_numpy[0, 0, :, :, 1] * 255
                model_temp[0, 0, :, :, 2] = out_numpy[0, 0, :, :, 2] * 255
                model_temp[0, 0, :, :, 3] = out_numpy[0, 0, :, :, 3] * 255

        model_screen[:, :, 0] = np.rint(model_temp_last[0, 0, :, :, frame])
        model_img = np.kron(model_screen, SCALE_ARR)
        block_screen[:, :, 0] = np.rint(0 * s0[0, 0, :, :, frame])
        block_screen[:, :, 1] = np.rint(0 * s0[0, 0, :, :, frame])
        block_screen[:, :, 2] = np.rint(255 * s0[0, 0, :, :, frame])
        block_img = np.kron(block_screen, SCALE_ARR) + target_screen + model_img
        new_surf = pygame.pixelcopy.make_surface(block_img.astype(np.uint8))
        screen.blit(new_surf, (0, 0))
        block_screen_last = block_screen


except StopIteration:
    pass

pygame.quit()
