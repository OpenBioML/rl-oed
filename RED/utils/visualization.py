import os

import matplotlib.pyplot as plt
import numpy as np


def plot_returns(returns, explore_rates=None, show=True, save_to_dir=None, conv_window=25):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_title("Return over time")

    if explore_rates is not None:
        ax2 = ax1.twinx()
        ax2.plot(np.repeat(explore_rates, len(returns) // len(explore_rates)), color="black", alpha=0.5, label="Explore Rate")
        ax2.set_ylabel("Explore Rate")
        ax2.legend(loc=1)

    ax1.plot(np.convolve(returns, np.ones(conv_window)/conv_window, mode="valid"), label="Return")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.legend(loc=2)

    if save_to_dir is not None:
        os.makedirs(save_to_dir, exist_ok=True)
        plt.savefig(os.path.join(save_to_dir, "returns.png"))

    if show:
        plt.show()
