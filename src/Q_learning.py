from dataclasses import dataclass
import numpy as np
import pygame

@dataclass
class Vector2:
    x: float
    y: float

class GridRender:
    def __init__(self, cell_width, cell_count_x, cell_count_y, start_pos ,line_thickness: float, screen: pygame.surface.Surface, reward_pos_grid: Vector2, blocks : iter):
        self.cell_width = cell_width
        self.cell_count_x = cell_count_x
        self.cell_count_y= cell_count_y
        self.line_thickness = line_thickness
        self.screen = screen
        self.start_pos = start_pos
        self.reward_pos_grid = reward_pos_grid
        self.blocks = blocks

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
agent_start_pos = Vector2(0,0)
grid_render = GridRender(cell_width=50, cell_count_x=cell_count_x, cell_count_y=cell_count_y, start_pos= Vector2(50, 100), line_thickness=4, screen=screen, reward_pos_grid=reward_pos, blocks=blocks)

while True:
    grid_render.render()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    pygame.display.update()