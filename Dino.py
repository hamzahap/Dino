import pygame
import random

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

# Set up fonts
FONT = pygame.font.SysFont(None, 48)

# Set up colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set up game loop
clock = pygame.time.Clock()
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and dino.y == GROUND_HEIGHT - 40:
                dino.y -= 100

    # Spawn new obstacles randomly
    if len(obstacles) < 5 and random.random() < 0.01:
        obstacle_height = random.randint(1, 3)
        obstacle = pygame.Rect(WINDOW_WIDTH, GROUND_HEIGHT - obstacle_height * 30, 20, obstacle_height * 30)
        obstacles.append(obstacle)

    # Move obstacles and check for collisions with the dino
    for obstacle in obstacles:
        obstacle.x -= 5
        if dino.colliderect(obstacle):
            text = FONT.render("Game Over", True, BLACK)
            WINDOW.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, WINDOW_HEIGHT // 2 - text.get_height() // 2))
            pygame.display.flip()
            pygame.time.wait(2000)
            running = False
        elif obstacle.right < 0:
            obstacles.remove(obstacle)

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
    pygame.display.flip()

    # Set the game clock
    clock.tick(60)

pygame.quit()
