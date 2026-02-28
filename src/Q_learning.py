import time
from dataclasses import dataclass
import random
from time import sleep
from tqdm.auto import tqdm
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
        self.agent_start_pos = Vector2(0,2)
        self.state = self.agent_start_pos

    def step(self, action) -> (int, float, bool):
        reward = -0.01
        is_done = False
        # Move if there is no border or wall
        direction_lut = [
            # action : 0 -> left
            Vector2(-1,0),
            # action : 1 -> up
            Vector2(0,-1),
            # action : 2] -> right
            Vector2(1, 0),
            # action : 3-> down
            Vector2(0, 1)
        ]
        direction = direction_lut[action]
        next_state = self.state + direction
        # Check if agent would move outside the border
        if next_state.x < 0 or next_state.x >= self.cell_count_x:
            next_state = self.state
        if next_state.y < 0 or next_state.y >= self.cell_count_y:
            next_state = self.state
            reward = -0.1
        # Check if a wall is there
        for block in self.blocks:
            if next_state == block:
                next_state = self.state
                reward = -0.1
        # Check if agent reacher target
        if next_state == self.reward_pos_grid:
            reward = 1
            is_done = True
            self.state = next_state
        self.state = next_state
        return (next_state.x*next_state.y), reward, is_done

    def reset(self):
        self.state = self.agent_start_pos
        return (self.state.x*self.state.y)
    def sample(self):
        """returns a random action"""
        action = random.randint(0,3)
        return action


    def render(self, episodes, n_finished, epsilon):
        screen.fill((0, 0, 0))
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
        # Render agent
        self.render_cell(self.state, (0,0,255))
        self.render_ui(episodes, n_finished, epsilon)
        pygame.display.update()

    def render_ui(self, episodes, n_finished, epsilon):
        font = pygame.font.SysFont("Comic Sans MS", 20)
        acc = n_finished/episodes *100
        label = font.render(f"Episodes: {episodes} | finished: {n_finished} | epsilon: {epsilon:.2f}", 1, (230,230,255))
        label2 = font.render(f"accuracy: {acc:.2f}%", 1, (230,230,255))
        self.screen.blit(label, (50, 300))
        self.screen.blit(label2, (50, 330))


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
#blocks = []
Env = ENV(cell_width=50, cell_count_x=cell_count_x, cell_count_y=cell_count_y, start_pos= Vector2(50, 100), line_thickness=4, screen=screen, reward_pos_grid=reward_pos, blocks=blocks)
# Q-Table
q_table = np.zeros((cell_count_x*cell_count_y, 4))
# Values
lr = 0.1 # Learning rate
gamma = 0.95 # value of future rewards
epsilon = 10 # exploration (random action) factor
min_epsilon = 0.05
epsilon_decay = 0.995
num_episodes = 100000
max_steps = 20
n_finished = 0
new_n_finished = 0
def choose_action(state):
    if random.uniform(0,1) <= epsilon:
        return Env.sample()
    else:
        return np.argmax(q_table[state,:])


for episode in tqdm(range(num_episodes)):
    #if episode % 10000 == 0:
       # print(f"Episode: {episode} | finished: {n_finished}")
    is_done = False
    state = Env.reset()
    for j in range(max_steps):

        # Rendering
        #Env.render(episode, n_finished, epsilon)
        if episode == 90000:
            min_epsilon = 0
            new_num_episodes = num_episodes - 90000
            new_n_finished = 0
        action = choose_action(state)
        next_state, reward, is_done = Env.step(action)
        old_value = q_table[state,action]
        next_max = np.max(q_table[next_state])

        q_table[state,action] = lr * old_value + (1-lr) * (reward + gamma * next_max)

        state = next_state
        if is_done:
            n_finished += 1
            new_n_finished += 1
            break

    epsilon = max(min_epsilon, epsilon*epsilon_decay)
acc = (new_n_finished/new_num_episodes)*100
print("left -------- up -------- right -------- down")
print(f"{q_table}")
print(f"\n accuracy: {acc:.2f}% | finished: {new_n_finished} | episodes: {new_num_episodes}")





