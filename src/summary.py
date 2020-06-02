import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from .config import *

def summary(rewards):
    plt.plot([i for i in range(1, EPISODES+1)], rewards)
    plt.title("Episode reward as a function of episode count")
    plt.xlabel("Episode count")
    plt.ylabel("Episode reward")
    plt.grid(True)
    plt.savefig(
        FIGURE_DIRECTORY
    )
    plt.show()

