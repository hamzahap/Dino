import numpy as np
import pygame
import sys
import random
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



# Set up the game parameters
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 150
GROUND_HEIGHT = 125
OBSTACLE_WIDTH = 20
OBSTACLE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
OBSTACLE_HEIGHTS = [1, 2, 3, 4, 5]
OBSTACLE_SPAWN_RATE = 100
OBSTACLE_SPEED = 5
DINO_SPEED = 8
JUMP_VELOCITY = 18
GRAVITY = 1
FONT_SIZE = 30
MEMORY_CAPACITY = 100000
BATCH_SIZE = 32
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.1
INITIAL_EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99
NUM_ACTIONS = 2
STATE_SIZE = 4
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# Initialize pygame
pygame.init()

# Set up the game window
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Dino Game")

# Set up the font
FONT = pygame.font.SysFont("Comic Sans MS", FONT_SIZE)

# Define the Dino class
class Dino:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = 0
        self.width = 30
        self.height = 50

    def jump(self):
        self.velocity = -JUMP_VELOCITY

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity

        if self.y > GROUND_HEIGHT:
            self.y = GROUND_HEIGHT
            self.velocity = 0

    def draw(self, surface):
        rect = pygame.Rect(self.x, self.y - self.height, self.width, self.height)
        pygame.draw.rect(surface, (0, 0, 0), rect)

# Define the Obstacle class
class Obstacle:
    def __init__(self, x, height):
        self.x = x
        self.y = GROUND_HEIGHT - OBSTACLE_WIDTH * height
        self.width = OBSTACLE_WIDTH
        self.height = OBSTACLE_WIDTH * height
        self.color = random.choice(OBSTACLE_COLORS)

    def update(self, speed):
        self.x -= speed

    def draw(self, surface):
        rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(surface, self.color, rect)

# Spawn a new obstacle
def spawn_obstacle(obstacles):
    height = random.choice(OBSTACLE_HEIGHTS)
    obstacle = Obstacle(WINDOW_WIDTH, height)
    obstacles.append(obstacle)

# Update the positions of the obstacles
def update_obstacles(obstacles, speed):
    for obstacle in obstacles:
        obstacle.update(speed)

# Remove any obstacles that are off the screen
def remove_offscreen_obstacles(obstacles):
    obstacles[:] = [obstacle for obstacle in obstacles if obstacle.x + obstacle.width > 0]

# Function to draw score to the screen
def draw_score(score):
    font = pygame.font.Font(None, 36)
    text = font.render("Score: " + str(score), True, BLACK)
    text_rect = text.get_rect()
    text_rect.centerx = WINDOW_WIDTH // 2
    text_rect.centery = 20
    WINDOW.blit(text, text_rect)

# Check for collisions between the dino and the obstacles
def check_collisions(dino, obstacles):
    for obstacle in obstacles:
        obstacle_height = obstacle.height // OBSTACLE_WIDTH
        if dino.y + dino.height > obstacle.y and dino.x + dino.width > obstacle.x and dino.x < obstacle.x + obstacle.width:
            return True
    return False

# Choose an action using the SARSA algorithm
def choose_action(model, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(NUM_ACTIONS)
    else:
        return np.argmax(model.predict(np.expand_dims(state, axis=0)))

# Update the Q-learning model using the SARSA algorithm
def replay(memory, model, batch_size, discount_factor, learning_rate):
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states = np.array([item[0] for item in batch])
    actions = np.array([item[1] for item in batch])
    rewards = np.array([item[2] for item in batch])
    next_states = np.array([item[3] for item in batch])
    next_actions = np.array([choose_action(model, state, epsilon=0) for state in next_states])
    q_values = model.predict(states)
    next_q_values = model.predict(next_states)
    targets = rewards + discount_factor * next_q_values[np.arange(batch_size), next_actions]
    q_values[np.arange(batch_size), actions] = (1 - learning_rate) * q_values[np.arange(batch_size), actions] + learning_rate * targets
    model.fit(states, q_values, verbose=0)

# Train the Q-learning model using SARSA
def train(model, memory, batch_size, discount_factor, learning_rate, epsilon, epsilon_min, epsilon_decay, obstacles):
    state = np.array([dino.y, (dino.y - dino.velocity) / 10, obstacles[0].x, obstacles[0].height])
    total_reward = 0
    done = False
    while not done:
        action = choose_action(model, state, epsilon)
        dino.jump() if action == 1 else None
        dino.update()
        update_obstacles(obstacles, OBSTACLE_SPEED)
        remove_offscreen_obstacles(obstacles)
        if len(obstacles) == 0:
            spawn_obstacle(obstacles)
        if check_collisions(dino, obstacles):
            reward = -10
            done = True
        else:
            reward = 1
        next_state = np.array([dino.y, (dino.y - dino.velocity) / 10, obstacles[0].x, obstacles[0].height]) if len(obstacles) > 0 else state
        memory.append((state, action, reward, next_state))
        replay(memory, model, batch_size, discount_factor, learning_rate)
        state = next_state
        total_reward += reward
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
    return total_reward

# Create the SARSA model using a deep neural network
def create_model():
    model = Sequential()
    model.add(Dense(64, input_shape=(STATE_SIZE,), activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(NUM_ACTIONS, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))
    return model


# Main game loop
def run_game():
    # Set up the Q-learning model
    model = create_model()
    memory = []
    epsilon = INITIAL_EPSILON
    total_reward = 0
    episode = 0
    obstacles = []

    while True:
        # Spawn a new obstacle if there are no obstacles
        if len(obstacles) == 0:
            spawn_obstacle(obstacles)

        # Set up the game
        global dino
        dino = Dino(50, GROUND_HEIGHT)

        # Run the game
        episode += 1
        reward = train(model, memory, BATCH_SIZE, DISCOUNT_FACTOR, LEARNING_RATE, epsilon, EPSILON_MIN, EPSILON_DECAY, obstacles)
        total_reward += reward

        # Update the obstacles
        for obstacle in obstacles:
            obstacle.move()
            if obstacle.off_screen():
                obstacles.remove(obstacle)

        # Draw the game window
        WINDOW.fill(WHITE)
        pygame.draw.rect(WINDOW, BLACK, (0, GROUND_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT - GROUND_HEIGHT))
        dino.draw(WINDOW)
        for obstacle in obstacles:
            obstacle.draw(WINDOW)
        draw_score(total_reward, episode, epsilon)
        pygame.display.update()

        # Check for collisions
        if check_collisions(dino, obstacles):
            epsilon = INITIAL_EPSILON
            total_reward = 0
            episode = 0
            obstacles = []
            dino.draw_death(WINDOW)
            pygame.display.update()
            time.sleep(1)
        
        # Update the display
        pygame.display.flip()



run_game()      
