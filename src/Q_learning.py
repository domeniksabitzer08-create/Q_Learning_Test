from dataclasses import dataclass
import numpy as np
import pygame

@dataclass
class Vector2:
    x: float
    y: float

class Grid:
    def __init__(self, cell_width, cell_count, start_pos ,line_thickness: float, screen: pygame.surface.Surface):
        self.cell_width = cell_width
        self.cell_count = cell_count
        self.line_thickness = line_thickness
        self.screen = screen
        self.start_pos = start_pos
    def render(self):
        for i in range(self.cell_count+1):
            # render horizontal lines
            line_pos_hor = Vector2(self.start_pos.x, self.start_pos.y + i * self.cell_width)
            line_pos_ver = Vector2(self.start_pos.x + i * self.cell_width, self.start_pos.y)
            self.render_line_hor(line_pos_hor)
            self.render_line_ver(line_pos_ver)



    def render_line_hor(self, pos: Vector2):
        line = pygame.Rect(pos.x, pos.y , self.cell_width * self.cell_count, self.line_thickness)
        pygame.draw.rect(self.screen, (255, 255, 255), line)

    def render_line_ver(self, pos: Vector2):
        line = pygame.Rect(pos.x, pos.y ,  self.line_thickness,self.cell_width * self.cell_count)
        pygame.draw.rect(self.screen, (255, 255, 255), line)


### MAIN PROGRAM STARTS HERE ###
pygame.init()
screen = pygame.display.set_mode((500, 500))
grid = Grid(cell_width=50, cell_count=6,start_pos= Vector2(50,100), line_thickness=4, screen=screen)
while True:
    grid.render()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    pygame.display.update()