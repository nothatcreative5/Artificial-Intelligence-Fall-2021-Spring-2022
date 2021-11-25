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

name = input("Please enter the name of the xml file you want to open.\nIf you don't have an xml file you can ignore "
             "this.\n")

try:
    xml = open(name + ".xml", "r")
    path = xml.read()
    doc = ET.fromstring(path)
    row = len(doc)
    col = len(doc[0])
    grid = np.zeros((row, col))
    grid_dir = np.full((row, col), 'N')
except FileNotFoundError:
    pass

pygame.init()


def read_from_xml():
    for i in range(row):
        for j in range(col):
            if doc[i][j].text == 'robot':
                x_start, y_start = i, j
                grid[i][j] = 1
            elif doc[i][j].text == 'Battery':
                grid[i][j] = 2
                x_goal, y_goal = i, j
            elif doc[i][j].text == 'obstacle':
                grid[i][j] = 3
            else:
                grid[i][j] = 0
    return grid, (x_start, y_start), (x_goal, y_goal)


map_type = int(input("To customize your own map, type 1.\nTo read the map from the xml file, type 2.\n"))
if map_type == 1:
    (x_start, y_start) = map(int,
                             input("Please enter the coordinates for the robot, separated by space.\n").split(' '))
    (x_goal, y_goal) = map(int, input("Please enter the coordinates for the battery, separated by space.\n").split(' '))
    (row, col) = map(int, input("Please enter the width and height of your map, separated by space.\n").split(' '))
    grid = np.zeros((row, col))
    grid_dir = np.full((row, col), 'N')
    grid[x_start][y_start] = 1
    grid[x_goal][y_goal] = 2
else:
    grid, (x_start, y_start), (x_goal, y_goal) = read_from_xml()


def save_as_XML(name, grid):
    map = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
    map += "<rows>\n"
    for i in range(row):
        map += "<row>\n"
        for j in range(col):
            if grid[i][j] == 1:
                map += "<cell>robot</cell>\n"
            elif grid[i][j] == 2:
                map += "<cell>Battery</cell>\n"
            elif grid[i][j] == 3:
                map += "<cell>obstacle</cell>\n"
            else:
                map += "<cell>empty</cell>\n"
        map += "</row>\n"
    map += "</rows>"
    text_file = open(name, "w")
    text_file.write(map)
    text_file.close()


# This sets the margin between each cell
MARGIN = 1

# This sets the WIDTH and HEIGHT of each grid location
WIDTH, HEIGHT = 40, 40

if WIDTH * col + (col + 1) * MARGIN > 720 or HEIGHT * row + (row + 1) * MARGIN > 720:
    HEIGHT = 720 // (row)
    WIDTH = 720 // (col)

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


def print_path():
    cur_x = x_goal
    cur_y = y_goal
    length = 0
    while cur_x != x_start or cur_y != y_start:
        length += 1
        grid[cur_x][cur_y] = 6
        print(cur_x, cur_y)
        cur_dir = grid_dir[cur_x][cur_y]
        if cur_dir == 'U':
            cur_x -= 1
        elif cur_dir == 'D':
            cur_x += 1
        elif cur_dir == 'R':
            cur_y += 1
        else:
            cur_y -= 1
        pygame.event.get()
        Draw()
    grid[x_goal][y_goal] = 2
    grid[x_start][y_start] = 1
    print(x_start, y_start)
    print('length = ', length + 1)
    print('number of explored rooms = ', len(explored_set) + 1)


def Search(h, scale):
    q = frontier.get()
    x, y, cost = q[1], q[2], q[3]
    if (x, y) in explored_set:
        return
    if x == x_goal and y == y_goal:
        print_path()
        frontier.queue.clear()
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
search_finish = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

def search_helper():
    message = "For A* with random heuristic, type 1.\nFor greedy best search, type 2.\nFor A* with Manhattan heuristic, type 3.\nfor uniform cost search, type 4.\n"
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
    elif type_of_search == 1:
        h = lambda x, y: np.random.randint(100,size = 1)[0]
        scale = 1
    while not frontier.empty():
        pygame.event.get()
        Search(h, scale)
        Draw()


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

            if search_finish and new_column == y_start and new_row == x_start:
                search_finish = False
                grid_dir = np.full((row, col), 'N')
                grid[np.logical_and(grid != 3, grid != 1, grid != 2)] = 0
                grid[x_start, y_start] = 1
                grid[x_goal][y_goal] = 2
            elif not search_finish and new_column == y_start and new_row == x_start:
                search_finish = True
                explored_set = []
                cost_set = np.full((row, col), -1)
                cost_set[x_start][y_start] = abs(x_start - x_goal) + abs(y_start - y_goal)
                search_helper()
                frontier.put((abs(x_start - x_goal) + abs(y_start - y_goal), x_start, y_start, 0))
            elif grid[new_row][new_column] == 3:
                grid[new_row][new_column] = 0
            elif new_row != x_goal or new_column != y_goal:
                grid[new_row][new_column] = 3

    Draw()

# Be IDLE friendly. If you forget this line, the program will 'hang'
# on exit.
pygame.quit()

save = int(input("Type 1 to save the map as an xml file.\nType 2 to skip this process.\n"))

if save == 1:
    save_as_XML(input("Please enter the file's name.\n") + ".xml", grid)
