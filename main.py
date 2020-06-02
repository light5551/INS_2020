import numpy as np
import gym
from gym.wrappers import Monitor
from lib.agent import Agent
from lib.summary import summary
from lib.config import *


def main():
    environment = gym.make(ENVIRONMENT)
    if RECORD:
        environment = Monitor(
            env=environment,
            directory=VIDEO_DIRECTORY,
            video_callable=lambda episode_id: True,
            force=True
        )
    environment.seed(0)
    np.random.seed(0)
    action_space = environment.action_space.n
    state_space = environment.observation_space.shape[0]
    agent = Agent(action_space, state_space)
    rewards = []
    for episode in range(EPISODES):
        state = environment.reset()
        state = np.reshape(state, (1, state_space))
        score = 0
        for _ in range(STEPS):
            environment.render()
            action = agent.act(state)
            next_state, reward, done, _ = agent.observe(environment, action)
            next_state = np.reshape(next_state, (1, state_space))
            observation = (state, action, reward, next_state, done)
            agent.remember(observation)
            state = next_state
            agent.learn()
            score += reward
            if done:
                print("Episode: {}/{}. Reward: {:.2f}".format(episode+1, EPISODES, score))
                break
        rewards.append(score)
        average_reward = np.mean(rewards[-100:])
        print("Average reward: {:.2f}\n".format(average_reward))
    environment.close()
    summary(rewards)


if __name__ == "__main__":
    main()

