import pygame
import time
import os
import random

pygame.init()

mouse_pic = pygame.image.load(str(os.getcwd())+'\mouse.jpg')

white = (255,255,255)
gray = (80,80,80)
black = (15,15,15)
red = (255,0,0)
green = (0,155,0)
blue = (0,0,155)

display_width = 800
display_height = 800

m, n = 5, 5

margin = 20
linew = 8

grid = [[1 for i in range(m)] for j in range(n)]
grid[3][0] = 0
grid[4][0] = 0
grid[0][2] = 0
grid[3][3] = 0
grid[1][3] = 0

gameDisplay = pygame.display.set_mode((display_width,display_height))

def pygame_window():
    gameExit = False
    perso = [0,0]

    count = 0

    while not gameExit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                gameExit = True

        count += 1

        if count == 15:
            if random.random() > 0.5:
                if random.random() > 0.5:
                    perso[0] += 1
                else:
                    perso[0] -= 1
                perso[0] = perso[0]%m
            else:
                if random.random() > 0.5:
                    perso[1] += 1
                else:
                    perso[1] -= 1
                perso[1] = perso[1]%n
            count = 0

        drawGrid(grid)
        draw_perso(perso)

        time.sleep(0.1)

        pygame.display.update()

    pygame.quit()

def draw_perso(perso):
    i, j = perso[0], perso[1]
    coords = [(display_width-2*margin)*i/m+margin+int(linew/2), (display_height-2*margin)*j/n+margin+int(linew/2), (display_width-2*margin)/m-linew ,(display_height-2*margin)/n-linew]
    coords = [int(i) for i in coords]

    resized_mouse_pic = pygame.transform.scale(mouse_pic, coords[2:])

    gameDisplay.blit(resized_mouse_pic, coords)

def drawGrid(grid):
    gameDisplay.fill(white)

    pygame.draw.rect(gameDisplay, gray, [margin-int(linew/2),margin-int(linew/2),display_width-2*margin+linew,display_height-2*margin+linew])

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                coords = [(display_width-2*margin)*i/m+margin+linew/2, (display_height-2*margin)*j/n+margin+linew/2, (display_width-2*margin)/m-linew ,(display_height-2*margin)/n-linew]
                coords = [int(i) for i in coords]
                pygame.draw.rect(gameDisplay, black, coords)

pygame_window()
