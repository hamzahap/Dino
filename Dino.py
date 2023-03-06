import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

# Initialize the font system
pygame.font.init()

# Set up fonts
FONT = pygame.font.SysFont(None, 48)

# Set up colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Define the neural network
class DinoNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size + 1, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Set up the neural network
input_size = 10 # the size of the input feature vector
hidden_size = 64 # the size of the hidden layer in the neural network
output_size = 1 # the size of the output (i.e. the probability of jumping)
model = DinoNet(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Set up the replay buffer
capacity = 100000 # the maximum capacity of the replay buffer
batch_size = 32 # the size of the batches used for training
replay_buffer = ReplayBuffer(capacity)

# Define the hyperparameters
gamma = 0.99 # the discount factor for future rewards
epsilon = 1.0 # the probability of taking a random action
epsilon_min = 0.01 # the minimum value of epsilon
epsilon_decay = 0.999 # the rate at which epsilon decays over time

# Set up the game loop
clock = pygame.time.Clock()
running = True
episode = 0
while running:
    # Reset the game if the dinosaur collides with an obstacle
    if any(obstacle.colliderect(dino) for obstacle in obstacles):
        # Update the score and print the result
        score_text = FONT.render("Score: " + str(score), True, BLACK)
        print(f"Episode {episode}: score={score}")
        episode += 1
        score = 0
        obstacles = []
        # Reset the dinosaur and obstacles
        dino = pygame.Rect(50, GROUND_HEIGHT - 40, 40, 40)
        obstacle_speed = 5
        obstacle_height_scale = 1

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get the state of the game
    state = np.array([
        dino.x, dino.y,
        *[obstacle.x for obstacle in obstacles],
        *[obstacle.y for obstacle in obstacles],
        *[obstacle_speed for obstacle in obstacles],
        score
    ])

    # Choose an action
    if np.random.rand() < epsilon:
        action = np.random.randint(2)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = model(state_tensor)
            action = int(action_tensor.squeeze().numpy() > 0.5)

    # Perform the action and update the game state
    if action == 1 and dino.y == GROUND_HEIGHT - 40:
        dino.y -= 100
        dino.y -= 150

    # Increase game speed over time
    if pygame.time.get_ticks() % 5000 == 0:
        obstacle_speed += 1
        obstacle_height_scale += 0.2

    # Spawn new obstacles randomly
    if len(obstacles) < 5 and random.random() < 0.01:
        obstacle_height = random.randint(1, 3) * obstacle_height_scale
        obstacle = pygame.Rect(WINDOW_WIDTH, GROUND_HEIGHT - obstacle_height * 15, 20, obstacle_height * 15)
        obstacles.append(obstacle)

    # Move obstacles and check for collisions with the dino
    for obstacle in obstacles:
        obstacle.x -= 5
        obstacle.x -= obstacle_speed
        if dino.colliderect(obstacle):
            reward = -1
            done = True
            replay_buffer.push(state, action, reward, None, done)
            break
        elif obstacle.right < 0:
            obstacles.remove(obstacle)
            score += 1
            reward = 1
            done = False
            replay_buffer.push(state, action, reward, None, done)

    # Move the dino back to the ground
    if dino.y < GROUND_HEIGHT - 40 and dino.y >= GROUND_HEIGHT - 140:
        dino.y += 5
    elif dino.y < GROUND_HEIGHT - 140:
        dino.y += 10

    # Get the next state of the game
    next_state = np.array([
        dino.x, dino.y,
        *[obstacle.x for obstacle in obstacles],
        *[obstacle.y for obstacle in obstacles],
        *[obstacle_speed for obstacle in obstacles],
        score
    ])

    # Update the Q-values using the Bellman equation
    if len(replay_buffer) > batch_size:
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)
        q_values = model(state_batch)
        next_q_values = model(next_state_batch)
        target_q_values = reward_batch + gamma * torch.max(next_q_values, dim=1)[0] * (1 - done_batch)
        q_values = q_values.gather(1, action_batch.long().unsqueeze(1)).squeeze(1)
        loss = criterion(q_values, target_q_values.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update epsilon
    epsilon = max(epsilon_min, epsilon_decay * epsilon)

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

pygame.quit()

