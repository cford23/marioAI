# DQN implementation code used from DQN Gridworld assignment file

from collections import deque
from pyboy import PyBoy, WindowEvent
import random
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

action_map = {
    'right': [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT],
    'left': [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT],
    'a': [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A],
}

action_set = {
    0: 'short right',
    1: 'short left',
    2: 'short a',
    3: 'medium right',
    4: 'medium left',
    5: 'medium a',
}

def passTime(pyboy, seconds):
    t_end = time.time() + seconds
    while time.time() < t_end:
        pyboy.tick()

def pressButton(pyboy, action):
    action = action.split()
    duration = action[0]
    button = action[1]

    if duration == 'short':
        holdTime = 0.1
    elif duration == 'medium':
        holdTime = 0.5
    
    pyboy.send_input(action_map[button][0])
    t_end = time.time() + holdTime
    while time.time() < t_end:
        pyboy.tick()
    pyboy.send_input(action_map[button][1])
    pyboy.tick()

class Agent():
    def __init__(self, epochs=5, gamma=0.9, epsilon=1.0, learning_rate=1e-3, batch_size=3200, mem_size=1000):
        self.epochs = epochs
        self.losses = []
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.replay = deque(maxlen=mem_size)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(320, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, len(action_set)) # one output for each action
        )
        self.state = []
        self.gamma = gamma
        self.epsilon = epsilon
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)


    def act(self, pyboy, state):
        qval = self.model(state)
        qval_np = qval.data.numpy()
        if random.random() < self.epsilon:
            action_np = np.random.randint(0, 3)
        else:
            action_np = np.argmax(qval_np)
        action = action_set[action_np]
        pressButton(pyboy, action)
        return action_np


    def train(self):
        pyboy = PyBoy('Super Mario Land.gb', game_wrapper=True, window_type='headless')
        pyboy.set_emulation_speed(0)
        mario = pyboy.game_wrapper()
        mario.start_game()
        avgFitnessScores = []
        print('Average Modified Fitness Scores')
        for i in range(self.epochs):
            mario.set_lives_left(0) # only allow 1 total life, more lives allowed if gained through playing
            prevModifiedFitnessScore = 0
            fitnessScores = []
            while not mario.game_over() and mario.world == (1, 1) and mario.time_left != 0: # ends game if agent dies or reaches end of level
                state1_np = np.asarray(mario.game_area()).flatten().astype(np.int16)
                state1 = torch.from_numpy(state1_np).float()
                action_np = self.act(pyboy, state1)
                
                state2_np = np.asarray(mario.game_area()).flatten().astype(np.int16)
                state2 = torch.from_numpy(state2_np).float()

                # Calculate reward for agent
                # Original fitness score: (lives_left * 10000) + (score + time_left * 10) + (_level_progress_max * 10)
                modifiedFitnessScore = (mario.score + mario.time_left * 10) + (mario._level_progress_max * 1000)
                if prevModifiedFitnessScore == 0:
                    reward = 0
                else:
                    reward = modifiedFitnessScore - prevModifiedFitnessScore

                fitnessScores.append(modifiedFitnessScore)
                prevModifiedFitnessScore = modifiedFitnessScore

                done = True if mario.world != (1, 1) or mario.game_over() or mario.time_left == 0 else False
                exp = (state1, action_np, reward, state2, done)
                self.replay.append(exp)
                state1 = state2

                if len(self.replay) > self.batch_size:
                    miniBatch = random.sample(self.replay, self.batch_size)
                    state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in miniBatch])
                    action_batch = torch.Tensor([a for (s1, a, r, s2, d) in miniBatch])
                    reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in miniBatch])
                    state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in miniBatch])
                    done_batch = torch.Tensor([d for (s1, a, r, s2, d) in miniBatch])

                    Q1 = self.model(state1_batch)
                    with torch.no_grad():
                        Q2 = self.model(state2_batch)
                    
                    Y = reward_batch + self.gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                    X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                    loss = self.loss_fn(X, Y.detach())
                    loss.backward()
                    self.losses.append(loss.item())
                    self.optimizer.step()
            
            if self.epsilon > 0.01:
                self.epsilon -= (1 / self.epochs)
            
            avgFitnessScore = round(np.average(fitnessScores))
            avgFitnessScores.append(avgFitnessScore)
            print(f'Epoch {i}/{self.epochs}: {avgFitnessScore}', end="\r")
            mario.reset_game()
        pyboy.stop()

        # create graph where x-axis is number of epochs and y-axis is the average modified fitness score
        plt.plot(np.arange(self.epochs), avgFitnessScores)
        plt.title('Agent Training Performance')
        plt.xlabel('Epoch')
        plt.ylabel('Avg Modified Fitness Score')
        plt.savefig(f'plots/{datetime.now()}_training.png')
    
    def save(self, modelFileName):
        torch.save(self.model.state_dict(), modelFileName)

    def test(self, lives):
        pyboy = PyBoy('Super Mario Land.gb', game_wrapper=True)
        mario = pyboy.game_wrapper()
        mario.start_game()
        mario.set_lives_left(lives - 1)

        try:
            # play until game over, level isn't 1-1 anymore (level has been completed), or time left == 0
            while not mario.game_over() and mario.world == (1, 1) and mario.time_left != 0:
                state_np = np.asarray(mario.game_area()).flatten().astype(np.int16)
                state = torch.from_numpy(state_np).float()
                self.act(pyboy, state)
            print('Done playing')
        finally:
            pyboy.stop()
            print('PyBoy has stopped')