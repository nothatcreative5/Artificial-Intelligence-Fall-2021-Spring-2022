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
    type_of_search = int(input("For A* non admissible heuristic, type 1.\nFor greedy best search, type 2.\nFor A*, type 3.\n"))
    if type_of_search == 3:
        h = lambda x, y: abs(x_goal - x) + abs(y_goal - y)
        scale = 1
    elif type_of_search == 2:
        h = lambda x, y: abs(x_goal - x) + abs(y_goal - y)
        scale = 0
    elif type_of_search == 1: