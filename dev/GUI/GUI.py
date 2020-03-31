import pygame
import time
import os
import random

# 1: empty tile, 2: small reward, 3: big reward, 0: unpassable terrain, -1: punishment
def show_trajectory(n_lin, n_col, piece, treasure, trap, obstacle, trajectory):

    grid = [[1 for i in range(n_col)] for j in range(n_lin)]

    treasure = [treasure]

    
    for i in piece:
        grid[i[0]][i[1]] = 2
    for i in treasure:
        grid[i[0]][i[1]] = 3
    for i in trap:
        grid[i[0]][i[1]] = -1
    for i in obstacle:
        grid[i[0]][i[1]] = 0

    pygame.init()

    display_width = 800
    display_height = 800

    gameDisplay = pygame.display.set_mode((display_width,display_height))

    gameExit = False
    traj_i = 0

    if trajectory == []:
        trajectory.append([0,0])

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

    white = (215,215,215)
    grey = (60,60,60)
    black = (15,15,15)

    m,n = len(grid),len(grid[0])

    cw = grid_pixel_coord(0,0,0,0,True)

    display.fill(grey)

    pygame.draw.rect(display, black, cw)

    mouse_pic = pygame.image.load(os.path.join(os.getcwd(),'GUI','pictures','mouse.png'))
    cheese_pic = pygame.image.load(os.path.join(os.getcwd(),'GUI','pictures','cheese.png'))
    morsel_pic = pygame.image.load(os.path.join(os.getcwd(),'GUI','pictures','morsel.png'))
    trap_pic = pygame.image.load(os.path.join(os.getcwd(),'GUI','pictures','trap.png'))

    pcoords = grid_pixel_coord(m, n, perso[0], perso[1])

    resized_mouse_pic = pygame.transform.scale(mouse_pic, pcoords[2:])
    resized_cheese_pic = pygame.transform.scale(cheese_pic, pcoords[2:])
    resized_morsel_pic = pygame.transform.scale(morsel_pic, pcoords[2:])
    resized_trap_pic = pygame.transform.scale(trap_pic, pcoords[2:])

    for i in range(m):
        for j in range(n):
            if grid[i][j]!= 0:
                coords = grid_pixel_coord(m, n, i, j)
                pygame.draw.rect(display, white, coords)
                if grid[i][j] == 2:
                    display.blit(resized_morsel_pic, coords[:2])
                elif grid[i][j] == 3:
                    display.blit(resized_cheese_pic, coords[:2])
                elif grid[i][j] == -1:
                    display.blit(resized_trap_pic, coords[:2])


    display.blit(resized_mouse_pic, pcoords[:2])


def grid_pixel_coord(m,n,i,j,w=False):
    display_width = 750
    display_height = 750
    margin = 50
    linew = 10

    if w:
        return [margin-int(linew/2),margin-int(linew/2),display_width-2*margin+linew,display_height-2*margin+linew]

    coords = [(display_width-2*margin)*i/m+margin+int(linew/2), (display_height-2*margin)*j/n+margin+int(linew/2), (display_width-2*margin)/m-linew ,(display_height-2*margin)/n-linew]

    return [int(i) for i in coords]


if __name__ == '__main__':
    n_lin = 5
    n_col = 5
    grid = [[1 for i in range(n_col)] for j in range(n_lin)]
    grid[2][1] = -1
    grid[4][2] = -1
    grid[2][2] = 2
    grid[4][3] = 3
    grid[0][3] = 0
    grid[2][0] = 0
    grid[3][3] = 0
    print(show_trajectory(grid, [[0,0],[0,1],[1,1],[1,2],[2,2],[3,2],[3,3]]))
