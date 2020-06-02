import random
import numpy as np
from collections import deque
from .model import Model
from .config import *


class Agent:
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.batch_size = BATCH_SIZE
        self.epsilon = 1.0
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.gamma = GAMMA
        self.memory = deque(maxlen=MEMORY_SIZE)
        print(action_space)
        print(state_space)
        self.model = Model(
            action_space,
            state_space,
            HIDDEN_NODES,
            HIDDEN_LAYERS,
            LEARNING_RATE
        ).build()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            actions = range(self.action_space-1)
            action = np.random.choice(actions)
        else:
            policy = self.model.predict(state)
            action = np.argmax(policy[0])
        return action

    def observe(self, environment, action):
        return environment.step(action)

    def remember(self, observation):
        self.memory.append(observation)

    def learn(self):
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states = np.array([i[0] for i in batch])
            actions = np.array([i[1] for i in batch])
            rewards = np.array([i[2] for i in batch])
            next_states = np.array([i[3] for i in batch])
            dones = np.array([i[4] for i in batch])
            states = np.squeeze(states)
            next_states = np.squeeze(next_states)
            action_state_value = self.model.predict_on_batch(states)
            next_action_state_value = rewards + self.gamma*np.amax(self.model.predict_on_batch(next_states), axis=1)*(1-dones)
            indices = np.array([i for i in range(self.batch_size)])
            action_state_value[[indices], [actions]] = next_action_state_value
            self.model.fit(states, action_state_value, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
