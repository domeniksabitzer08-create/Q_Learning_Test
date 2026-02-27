from dataclasses import dataclass
import random
import numpy as np
import pygame
from torch.distributed.argparse_util import env


@dataclass
class Vector2:
    x: float
    y: float
    def __add__(self, other):
        # Vector + Vector
        if isinstance(other, Vector2):
            return Vector2(self.x + other.x, self.y + other.y)

class ENV:
    def __init__(self, cell_width, cell_count_x, cell_count_y, start_pos ,line_thickness: float, screen: pygame.surface.Surface, reward_pos_grid: Vector2, blocks : iter):
        self.cell_width = cell_width
        self.cell_count_x = cell_count_x
        self.cell_count_y= cell_count_y
        self.line_thickness = line_thickness
        self.screen = screen
        self.start_pos = start_pos
        self.reward_pos_grid = reward_pos_grid
        self.blocks = blocks
        self.agent_start_pos = Vector2(0,0)
        self.state = self.agent_start_pos

    def step(self, action) -> (Vector2, int, bool):
        reward = -1
        is_done = False
        # Move if there is no border or wall
        direction_lut = [
            # action : [1,0,0,0] -> left
            Vector2(-1,0),
            # action : [0,1,0,0] -> up
            Vector2(0,-1),
            # action : [0,0,1,0] -> right
            Vector2(1, 0),
            # action : [0,0,0,1] -> down
            Vector2(0, 1)
        ]
        direction = direction_lut[np.argmax(action)]
        next_state = self.state + direction
        # Check if agent would move outside the border
        if next_state.x < 0 or next_state.x >= self.cell_count_x:
            next_state = self.state
        if next_state.y < 0 or next_state.y >= self.cell_count_y:
            next_state = self.state
        # Check if a wall is there
        for block in self.blocks:
            if next_state == block:
                next_state = self.state
        # Check if agent reacher target
        if next_state == self.reward_pos_grid:
            reward = 0
            is_done = True
        self.state = next_state
        return next_state, reward, is_done

    def reset(self) -> Vector2:
        self.state = self.agent_start_pos
        return self.state
    def sample(self):
        """returns a random action"""
        action = [0,0,0,0]
        rnd_sample = np.random.sample(4).tolist()
        action[np.argmax(rnd_sample)] = 1
        return action


    def render(self):
        # render vertical lines
        for i in range(self.cell_count_x+1):
            line_pos_ver = Vector2(self.start_pos.x + i * self.cell_width, self.start_pos.y)
            self.render_line_ver(line_pos_ver)
        # render horizontal lines
        for i in range(self.cell_count_y+1):
            line_pos_hor = Vector2(self.start_pos.x, self.start_pos.y + i * self.cell_width)
            self.render_line_hor(line_pos_hor)
        # Render reward cell
        self.render_cell(self.reward_pos_grid, (0,100,0))
        # Render blocks
        for pos in self.blocks:
            self.render_cell(pos, (100,100,100))

    def render_line_hor(self, pos: Vector2):
        line = pygame.Rect(pos.x, pos.y , self.cell_width * self.cell_count_x + self.line_thickness, self.line_thickness)
        pygame.draw.rect(self.screen, (255, 255, 255), line)

    def render_line_ver(self, pos: Vector2):
        line = pygame.Rect(pos.x, pos.y ,  self.line_thickness,self.cell_width * self.cell_count_y + self.line_thickness)
        pygame.draw.rect(self.screen, (255, 255, 255), line)

    def render_cell(self, pos_grid: Vector2, color = (0,100,0)):
        pos = self.grid_to_screen_pos(pos_grid)
        rect = pygame.Rect(pos.x + self.line_thickness , pos.y + self.line_thickness, self.cell_width - self.line_thickness , self.cell_width - self.line_thickness)
        pygame.draw.rect(self.screen, color, rect)

    def grid_to_screen_pos(self, pos: Vector2):
        return Vector2(pos.x * self.cell_width + self.start_pos.x , pos.y * self.cell_width + self.start_pos.y)

### MAIN PROGRAM STARTS HERE ###
pygame.init()
screen = pygame.display.set_mode((500, 500))
# Grid setup
cell_count_x, cell_count_y = 5, 3
reward_pos = Vector2(4,2) # in grid coordinates
blocks = [Vector2(2,2), Vector2(2,1)]
Env = ENV(cell_width=50, cell_count_x=cell_count_x, cell_count_y=cell_count_y, start_pos= Vector2(50, 100), line_thickness=4, screen=screen, reward_pos_grid=reward_pos, blocks=blocks)
# Q-Table
q_table = np.zeros((cell_count_x, cell_count_y))
# Values
lr = 0.1 # Learning rate
gamma = 0.95 # value of future rewards
epsilon = 1 # exploration (random action) factor
epsilon_decay = 0.95
num_episodes = 1000
max_steps = 20

def choose_action(state):
    if random.uniform(0,1) <= epsilon:
        return Env.sample()
    else:
        return np.argmax(q_table[state])

while True:
    for i in range(num_episodes):
        is_done = False
        state = Env.reset()
        for j in range(max_steps):
            action = choose_action(state)
            next_state, reward, is_done = Env.step(action)

            old_value = q_table[state,action]
            next_max = np.max(q_table[next_state])

            q_table[state,action] = lr * old_value + (1-lr) * (reward + gamma * next_max)




    Env.render()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    pygame.display.update()