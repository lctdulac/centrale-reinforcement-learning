import pygame
import time
import os
import random

# 1: empty tile, 2: small reward, 3: big reward, 0: unpassable terrain, -1: punishment
def show_trajectory(n_lin, n_col, trajectory):

    grid = [[1 for i in range(n_col)] for j in range(n_lin)]

    pygame.init()

    display_width = 800
    display_height = 800

    gameDisplay = pygame.display.set_mode((display_width,display_height))

    gameExit = False
    traj_i = 0

    draw_grid(gameDisplay, grid, trajectory[traj_i])
    pygame.display.update()

    while not gameExit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True
            if event.type == pygame.MOUSEBUTTONDOWN:
                traj_i+= 1
                if traj_i >= len(trajectory):
                    gameExit = True
                else:
                    draw_grid(gameDisplay, grid, trajectory[traj_i])
                    pygame.display.update()

    pygame.quit()

    return True


def draw_grid(display, grid, perso):

    white = (155,155,155)
    grey = (60,60,60)
    black = (15,15,15)

    m,n = len(grid),len(grid[0])

    cw = grid_pixel_coord(0,0,0,0,True)

    display.fill(white)

    pygame.draw.rect(display, grey, cw)

    for i in range(m):
        for j in range(n):
            coords = grid_pixel_coord(m, n, i, j)
            pygame.draw.rect(display, black, coords)

    mouse_pic = pygame.image.load(str(os.getcwd())+'\pictures\mouse.png')

    coords = grid_pixel_coord(m, n, perso[0], perso[1])

    resized_mouse_pic = pygame.transform.scale(mouse_pic, coords[2:])

    display.blit(resized_mouse_pic, coords[:2])


def grid_pixel_coord(m,n,i,j,w=False):
    display_width = 800
    display_height = 800
    margin = 50
    linew = 10

    if w:
        return [margin-int(linew/2),margin-int(linew/2),display_width-2*margin+linew,display_height-2*margin+linew]

    coords = [(display_width-2*margin)*i/m+margin+int(linew/2), (display_height-2*margin)*j/n+margin+int(linew/2), (display_width-2*margin)/m-linew ,(display_height-2*margin)/n-linew]

    return [int(i) for i in coords]


if __name__ == '__main__':
    grid = [[1 for i in range(5)] for j in range(4)]
    print(show_trajectory(grid, [[0,0],[1,0],[2,0],[2,1],[2,2],[2,3],[3,3]]))
