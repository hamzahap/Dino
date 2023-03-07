import pygame
import numpy as np
import sys
import os
import tensorflow as tf
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
obstacles = [(i * 300 + 600, GROUND_HEIGHT - 60) for i in range(10)] # List of obstacle positions
obstacle_speed = 5
score = 0

# Set up fonts
FONT = pygame.font.SysFont(None, 48)

# Set up colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set up deep Q-learning parameters
GAMMA = 0.9
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MEMORY_CAPACITY = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.001
STATE_SIZE = 15 # Number of features in the state
ACTION_SIZE = 2
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_dim=STATE_SIZE, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(ACTION_SIZE, activation=None)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
memory = []

# Helper function to preprocess the state
def preprocess_state(dino, obstacles):
    state = []
    region_size = WINDOW_WIDTH // 10
    for i in range(10):
        region_start = i * region_size
        region_end = (i + 1) * region_size
        if any(region_start <= obs[0] < region_end for obs in obstacles):
            state.append(1)
        else:
            state.append(0)
    state.append(1 if dino.y < GROUND_HEIGHT - 140 else 0)
    state.append(obstacles[0][0] - dino.x)
    state.append(obstacle_speed)
    state.append(score)
    state.append(obstacles[1][0] - dino.x)
    return np.array(state)

# Helper function to select an action using epsilon-greedy policy
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(ACTION_SIZE)
    else:
        return np.argmax(model.predict(np.array([state])))

# Helper function to add a transition to memory
def add_to_memory(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))
    if len(memory) > MEMORY_CAPACITY:
        memory.pop(0)

# Helper function to sample a minibatch from memory
def sample_minibatch():
    minibatch = random.sample(memory, BATCH_SIZE)
    states = []
    targets = []
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target += GAMMA * np.max(model.predict(np.array([next_state]))[0])
        target_vec = model.predict(np.array([state]))[0]
        target_vec[action] = target
        states.append(state)
        targets.append(target_vec)
    return np.array(states), np.array(targets)

# Set up game loop
clock = pygame.time.Clock()
running = True
epsilon = EPSILON_START
state = preprocess_state(dino, obstacles)
model_loaded = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Select an action using epsilon-greedy policy
    if model_loaded:
        action = select_action(state, epsilon)
    else:
        model.load_weights('model.h5')
        action = select_action(state, 0.0)
        model_loaded = True

    # Take the action and observe the next state and reward
    if action == 1 and dino.y == GROUND_HEIGHT - 40:
        dino.y -= 100
        dino.y -= 150

    # Move the dino back to the ground
    if dino.y < GROUND_HEIGHT - 40 and dino.y >= GROUND_HEIGHT - 140:
        dino.y += 5
    elif dino.y < GROUND_HEIGHT - 140:
        dino.y += 10

    new_state = preprocess_state(dino, obstacles)
    if obstacles[0][0] - dino.x < 40:
        reward = -10
        done = True
    elif dino.colliderect(pygame.Rect(obstacles[0][0], obstacles[0][1], 40, 60)):
        reward = -10
        done = True
    elif obstacles[0][0] - dino.x < 100:
        reward = 1
        done = False
    else:
        reward = 0
        done = False

    # Add the transition to memory
    add_to_memory(state, action, reward, new_state, done)
    state = new_state

    # Update the model using a minibatch of transitions from memory
    if len(memory) >= BATCH_SIZE:
        states, targets = sample_minibatch()
        model.train_on_batch(states, targets)

    # Decay epsilon
    if epsilon > EPSILON_END:
        epsilon -= EPSILON_DECAY

    # Move obstacles and check for collisions with the dino
    for i in range(len(obstacles)):
        obstacles[i] = (obstacles[i][0] - obstacle_speed, obstacles[i][1])
        if obstacles[i][0] < -40:
            obstacles[i] = (obstacles[(i - 1) % len(obstacles)][0] + np.random.randint(300, 500), GROUND_HEIGHT - 60)
        if dino.colliderect(pygame.Rect(obstacles[i][0], obstacles[i][1], 40, 60)):
            text = FONT.render("Game Over", True, BLACK)
            WINDOW.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, WINDOW_HEIGHT // 2 - text.get_height() // 2))
            pygame.display.flip()
            pygame.time.wait(2000)
            running = False
        if obstacles[i][0] < dino.x and (i + 1) % len(obstacles) != 0:
            next_obstacle = obstacles[(i + 1) % len(obstacles)]
            if next_obstacle[0] - dino.x < 200:
                reward = -10
                done = True

    # Increment the score if the dino has passed an obstacle
    if obstacles[0][0] - dino.x < 0:
        score += 1

    WINDOW.fill(WHITE)
    pygame.draw.rect(WINDOW, BLACK, pygame.Rect(0, GROUND_HEIGHT, WINDOW_WIDTH, WINDOW_HEIGHT - GROUND_HEIGHT))
    pygame.draw.rect(WINDOW, (255, 0, 0), dino)
    for obstacle in obstacles:
        pygame.draw.rect(WINDOW, (0, 0, 255), pygame.Rect(obstacle[0], obstacle[1], 40, 60))

    score_text = FONT.render("Score: " + str(score), True, BLACK)
    WINDOW.blit(score_text, (10, 10))

    # Draw the AI state information
    info_text = FONT.render("AI State", True, BLACK)
    WINDOW.blit(info_text, (10, 50))

    ai_state = preprocess_state(dino, obstacles)
    info_text = FONT.render("AI Input: " + str(ai_state), True, BLACK)
    WINDOW.blit(info_text, (10, 80))

    if model_loaded:
        ai_action = select_action(ai_state, 0)
        info_text = FONT.render("AI Action: " + str(ai_action), True, BLACK)
        WINDOW.blit(info_text, (10, 110))

    pygame.display.flip()

    # Set the game clock
    clock.tick(60)

    # End the game if all obstacles have been conquered
    if score == len(obstacles):
        text = FONT.render("You Win!", True, BLACK)
        WINDOW.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, WINDOW_HEIGHT // 2 - text.get_height() // 2))
        pygame.display.flip()
        pygame.time.wait(2000)
        running = False

if model_loaded:
    model.save_weights('model.h5')

pygame.quit()
os.execv(sys.executable, [sys.executable] + sys.argv)