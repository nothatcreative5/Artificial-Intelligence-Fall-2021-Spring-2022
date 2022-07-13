import xml.etree.ElementTree as ET
from queue import PriorityQueue
import pygame
import numpy as np

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
VIOLET = (255, 0, 190)
PEACHPUFF = (255, 218, 185)
GOLD = (255, 215, 0)

# Directions
dRow = [1, 0, -1, 0]
dCol = [0, 1, 0, -1]
directions = ['U', 'L', 'D', 'R']

name = input("Please enter the name of the xml file you want to open.\n")

xml = open(name + ".xml", "r")
path = xml.read()
doc = ET.fromstring(path)
row = len(doc)
col = len(doc[0])
grid, data = np.zeros((row, col)), np.zeros((row, col))
grid_dir = np.full((row, col), 'N')

pygame.init()
total_explored = 0
total_walks = 0


def read_from_xml():
    global x_start, y_start, x_goal, y_goal
    for i in range(row):
        for j in range(col):
            if doc[i][j].text == 'robot':
                x_start, y_start = i, j
                grid[i][j] = 1
                data[i][j] = 1
            elif doc[i][j].text == 'Battery':
                data[i][j] = 2
                grid[i][j] = 2
                x_goal, y_goal = i, j
            elif doc[i][j].text == 'obstacle':
                data[i][j] = 3
            else:
                grid[i][j] = 0
                data[i][j] = 0
    return grid, (x_start, y_start), (x_goal, y_goal)


grid, (x_start, y_start), (x_goal, y_goal) = read_from_xml()
Const_start_x, Const_start_y = x_start, y_start

# This sets the margin between each cell
MARGIN = 1

# This sets the WIDTH and HEIGHT of each grid location
WIDTH, HEIGHT = 40, 40

if WIDTH * col + (col + 1) * MARGIN > 720 or HEIGHT * row + (row + 1) * MARGIN > 720:
    HEIGHT = 720 // row
    WIDTH = 720 // col

frontier = PriorityQueue()
explored_set = []
cost_set = np.full((row, col), -1)
cost_set[x_start][y_start] = abs(x_start - x_goal) + abs(y_start - y_goal)
frontier.put((abs(x_start - x_goal) + abs(y_start - y_goal), x_start, y_start, 0))


def isValid(x_cord, y_cord, cost):
    if x_cord >= row or x_cord < 0 or y_cord >= col or y_cord < 0 or ((x_cord, y_cord) in explored_set) \
            or grid[x_cord][y_cord] == 3 or (cost_set[x_cord][y_cord] < cost and cost_set[x_cord][y_cord] != -1):
        return False
    return True


def reverse_path():
    global grid_dir
    temp_grid = np.full((row, col), 'N')
    cur_x = x_goal
    cur_y = y_goal
    while cur_x != x_start or cur_y != y_start:
        cur_dir = grid_dir[cur_x][cur_y]
        print(cur_x,cur_y)
        if cur_dir == 'U':
            cur_x -= 1
            temp_grid[cur_x][cur_y] = 'D'
        elif cur_dir == 'D':
            cur_x += 1
            temp_grid[cur_x][cur_y] = 'U'
        elif cur_dir == 'R':
            cur_y += 1
            temp_grid[cur_x][cur_y] = 'L'
        else:
            cur_y -= 1
            temp_grid[cur_x][cur_y] = 'R'
    grid[cur_x][cur_y] = 6
    grid_dir = np.copy(temp_grid)


def print_path():
    global x_start, y_start, grid_dir, explored_set, cost_set, total_walks, total_explored
    cur_x = x_start
    cur_y = y_start
    total_explored += len(explored_set)
    while cur_x != x_goal or cur_y != y_goal:
        temp_x, temp_y = cur_x, cur_y
        total_walks += 1
        grid[temp_x][temp_y] = 6
        cur_dir = grid_dir[cur_x][cur_y]
        if cur_dir == 'U':
            cur_x -= 1
        elif cur_dir == 'D':
            cur_x += 1
        elif cur_dir == 'R':
            cur_y += 1
        else:
            cur_y -= 1
        grid[cur_x][cur_y] = 1
        if data[cur_x][cur_y] == 3:
            x_start, y_start = temp_x, temp_y
            grid[cur_x][cur_y] = 3
            reset(True)
            return

        pygame.event.get()
        Draw()
    grid[x_goal][y_goal] = 2
    grid[x_start][y_start] = 1
    print('length = ', total_walks + 1)
    print('number of explored rooms = ', total_explored + 1)
    frontier.queue.clear()


def Draw():
    # Set the screen background
    screen.fill(BLACK)
    for i in range(row):
        for j in range(col):
            color = WHITE
            if grid[i][j] == 1:
                color = GREEN
            elif grid[i][j] == 2:
                color = RED
            elif grid[i][j] == 3:
                color = BLUE
            elif grid[i][j] == 4:
                color = VIOLET
            elif grid[i][j] == 5:
                color = PEACHPUFF
            elif grid[i][j] == 6:
                color = GOLD
            pygame.draw.rect(screen,
                             color,
                             [(MARGIN + WIDTH) * j + MARGIN,
                              (MARGIN + HEIGHT) * i + MARGIN,
                              WIDTH,
                              HEIGHT])

    # Limit to 120 frames per second
    clock.tick(120)

    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()


# Set the HEIGHT and WIDTH of the screen
WINDOW_SIZE = [WIDTH * col + (col + 1) * MARGIN, HEIGHT * row + (row + 1) * MARGIN]
screen = pygame.display.set_mode(WINDOW_SIZE)

# Set title of screen
pygame.display.set_caption("Search Algorithms")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()


def reset(for_search=False):
    global grid, x_start, y_start, explored_set, grid_dir, cost_set
    if not for_search:
        grid = np.zeros((row, col))
        x_start, y_start = Const_start_x, Const_start_y
    grid[np.logical_and(grid != 3, grid != 1, grid != 2)] = 0
    grid[x_start][y_start], grid[x_goal][y_goal] = 1, 2
    grid_dir = np.full((row, col), 'N')
    explored_set = []
    frontier.queue.clear()
    frontier.put((abs(x_start - x_goal) + abs(y_start - y_goal), x_start, y_start, 0))
    cost_set = np.full((row, col), -1)
    cost_set[x_start][y_start] = abs(x_start - x_goal) + abs(y_start - y_goal)


def Search(h, scale):
    q = frontier.get()
    x, y, cost = q[1], q[2], q[3]
    if (x, y) in explored_set:
        return
    if x == x_goal and y == y_goal:
        reverse_path()
        print_path()
        return
    for i in range(4):
        x += dRow[i]
        y += dCol[i]
        if isValid(x, y, h(x, y) + (cost + 1) * scale):
            neighbor = ((cost + 1) * scale + h(x, y), x, y, cost + 1)
            cost_set[x][y] = (cost + 1) * scale + h(x, y)
            grid_dir[x][y] = directions[i]
            frontier.put(neighbor)
            grid[x][y] = 5
        x = q[1]
        y = q[2]
        grid[x][y] = 4
    explored_set.append((q[1], q[2]))


def search_helper():
    global total_walks, total_explored
    message = "For A* with random heuristic, type 1.\nFor greedy best search, type 2.\nFor A* with Manhattan " \
              "heuristic, type 3.\nfor uniform cost search, type 4.\n "
    type_of_search = int(input(message))
    if type_of_search == 4:
        h = lambda x, y: 0
        scale = 1
    elif type_of_search == 3:
        h = lambda x, y: abs(x_goal - x) + abs(y_goal - y)
        scale = 1
    elif type_of_search == 2:
        h = lambda x, y: abs(x_goal - x) + abs(y_goal - y)
        scale = 0
    else:
        h = lambda x, y: np.random.randint(100, size=1)[0]
        scale = 1
    while not frontier.empty():
        pygame.event.get()
        Search(h, scale)
        Draw()
    total_walks, total_explored = 0, 0


# -------- Main Program Loop -----------
while not done:
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # User clicks the mouse. Get the position
            pos = pygame.mouse.get_pos()
            # Change the x/y screen coordinates to grid coordinates
            new_column = pos[0] // (WIDTH + MARGIN)
            new_row = pos[1] // (HEIGHT + MARGIN)

            if new_column == y_start and new_row == x_start:
                reset()
                search_helper()

    Draw()

# Be IDLE friendly. If you forget this line, the program will 'hang'
# on exit.
pygame.quit()