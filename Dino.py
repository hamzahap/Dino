import pygame
import random
import numpy as np
import sys

# Set the random seed for reproducibility
random.seed(42)

pygame.init()

# Set up the game window
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 400
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Dino Game")

# Set up game variables
GROUND_HEIGHT = 350
dino = pygame.Rect(50, GROUND_HEIGHT - 40, 40, 40)
obstacles = []
obstacle_speed = 5
obstacle_height_scale = 1
score = 0

# Set up fonts
FONT = pygame.font.SysFont(None, 48)

# Set up colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set up Q-learning parameters
GAMMA = 0.9
LEARNING_RATE = 0.1
EPSILON = 0.1
Q_TABLE = np.zeros((3, 2))

# Helper function to discretize the state
def discretize_state(dino, obstacles):
    dino_pos = dino.y
    obstacle_pos = 0
    if len(obstacles) > 0:
        obstacle_pos = obstacles[0].x
    if obstacle_pos < dino.x:
        obstacle_pos = 0
    state = 0
    if dino_pos < GROUND_HEIGHT - 140:
        state = 2
    elif obstacle_pos > 0:
        state = 1
    return state

# Save the Q-table to a file at the end of the game
np.savetxt('qtable.txt', Q_TABLE)

# Load the Q-table from the file at the beginning of the next game
Q_TABLE = np.loadtxt('qtable.txt')

# Set up game loop
clock = pygame.time.Clock()
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and dino.y == GROUND_HEIGHT - 40:
                dino.y -= 100
                dino.y -= 150

    # Get the current state
    state = discretize_state(dino, obstacles)

    # Choose an action
    if random.random() < EPSILON:
        action = random.randint(0, 1)
    else:
        action = np.argmax(Q_TABLE[state])

    # Perform the action
    if action == 0:
        pass
    elif action == 1:
        if dino.y == GROUND_HEIGHT - 40:
            dino.y -= 100
            dino.y -= 150

    # Increase game speed over time
    if pygame.time.get_ticks() % 5000 == 0:
        obstacle_speed += 1
        obstacle_height_scale += 0.2

    # Spawn new obstacles randomly
    if len(obstacles) < 5 and random.random() < 0.01:
        obstacle_height = random.randint(1, 3)
        obstacle_width = random.randint(20, 40)
        obstacle = pygame.Rect(WINDOW_WIDTH, GROUND_HEIGHT - obstacle_height * 30, obstacle_width, obstacle_height * 30)
        obstacles.append(obstacle)

    # Move obstacles and check for collisions with the dino
    for obstacle in obstacles:
        obstacle.x -= 5
        obstacle.x -= obstacle_speed
        if dino.colliderect(obstacle):
            # Update the Q-table with the final reward
            Q_TABLE[state][action] = (1 - LEARNING_RATE) * Q_TABLE[state][action] + \
                LEARNING_RATE * (reward + GAMMA * np.max(Q_TABLE[new_state]))

            # Reset the game variables
            dino = pygame.Rect(50, GROUND_HEIGHT - 40, 40, 40)
            obstacles = []
            obstacle_speed = 5
            obstacle_height_scale = 1
            score = 0

            # Set the initial state and action
            state = discretize_state(dino, obstacles)
            if random.random() < EPSILON:
                action = random.randint(0, 1)
            else:
                action = np.argmax(Q_TABLE[state])

            # Draw the game window
            WINDOW.fill(WHITE)
            pygame.draw.rect(WINDOW, BLACK, pygame.Rect(0, GROUND_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT - GROUND_HEIGHT))
            pygame.draw.rect(WINDOW, (255, 0, 0), dino)
            for obstacle in obstacles:
                pygame.draw.rect(WINDOW, (0, 0, 255), obstacle)
            score_text = FONT.render("Score: " + str(score), True, BLACK)
            WINDOW.blit(score_text, (10, 10))
            pygame.display.flip()

            # Wait for 2 seconds before resuming the game
            pygame.time.wait(2000)

            # Set the game clock
            clock.tick(60)

        elif obstacle.right < 0:
            obstacles.remove(obstacle)
            score += 1

    # Get the new state and reward
    new_state = discretize_state(dino, obstacles)
    if new_state == 0:
        reward = 1
    else:
        reward = 0

    # Update the Q-table
    Q_TABLE[state][action] = (1 - LEARNING_RATE) * Q_TABLE[state][action] + \
        LEARNING_RATE * (reward + GAMMA * np.max(Q_TABLE[new_state]))

    # Move the dino back to the ground
    if dino.y < GROUND_HEIGHT - 40 and dino.y >= GROUND_HEIGHT - 140:
        dino.y += 5
    elif dino.y < GROUND_HEIGHT - 140:
        dino.y += 10

    # Draw the game window
    WINDOW.fill(WHITE)
    pygame.draw.rect(WINDOW, BLACK, pygame.Rect(0, GROUND_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT - GROUND_HEIGHT))
    pygame.draw.rect(WINDOW, (255, 0, 0), dino)
    for obstacle in obstacles:
        pygame.draw.rect(WINDOW, (0, 0, 255), obstacle)
    score_text = FONT.render("Score: " + str(score), True, BLACK)
    WINDOW.blit(score_text, (10, 10))
    pygame.display.flip()

    # Set the game clock
    clock.tick(60)

# Save the Q-table to a file
np.savetxt('qtable.txt', Q_TABLE)

pygame.quit()