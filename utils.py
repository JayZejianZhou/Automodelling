import time

import torch
import numpy as np
from collections import namedtuple, deque
import random
import matplotlib.pyplot as plt

# store the model
def save_model(net, name, save_mode=1):
    # net - the NN to be saved
    # name - the file name
    # save_mode - 0 - save the entire model, 1 - save only params, 2 - save only params but with full path
    if save_mode == 0:
        torch.save(net, name+".pkl")
    elif save_mode == 1:
        torch.save(net.state_dict(), name+"_params"+".pkl") # save params only, not NN structure
    elif save_mode == 2:
        torch.save(net.state_dict(), name)  # save params only, not NN structure
    print("save model to "+name)


# read in model
def load_model(name, net=None, load_mode=1):
    # net - the NN to load the data, none if it's full save
    # name - the file name
    # load_mode - 0 - load the entire model, 1 - load only params, 2 - load only params but with full path
    if load_mode == 0:
        net = torch.load(name + ".pkl")
    elif load_mode == 1:
        net.load_state_dict(torch.load(name + "_params" + ".pkl"))  # load params only, not NN structure
    elif load_mode == 2:
        net.load_state_dict(torch.load(name))
    return net

# plot the cost function
def plot_cost(net, length, width):
    costs = np.zeros([length, width])
    for i in range(length):
        for j in range(width):
            costs[i, j] = net(torch.tensor([i, j]).type(torch.FloatTensor)).data
    return costs

def plot_ep_reward_history(total_rewards):
    # After training, you can visualize or analyze the agent's Q-table or policy
    fig, ax = plt.subplots()
    ax.plot(total_rewards)
    ax.set_title("Total Rewards Over Episodes")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Total Reward")
    plt.show()

def check_GPU_available():
    if torch.cuda.is_available():
        print("GPU is available!")
    else:
        print("GPU is not available.")


# this class manages the figures
class FigureManager:
    def __init__(self):
        self.figures = {}
    # fig_id is the figure name
    def create_figure(self, fig_id, num_axes=1, title=None):
        if fig_id in self.figures:
            print(f"Warning: Overwriting figure with ID {fig_id}")

        fig, ax = plt.subplots(num_axes)
        if title:
            fig.suptitle(title)

        # Ensure axes are always in list format even if there's just one axis
        if num_axes == 1:
            ax = [ax]

        self.figures[fig_id] = {'figure': fig, 'axis': ax}
        return self.figures[fig_id]

    def has_figure(self, fig_to_check):
        """Check if a figure object exists in the manager."""
        return fig_to_check in [fig_data['figure'] for fig_data in self.figures.values()]

    # Register an existing figure
    def register_figure(self, fig_id, fig, ax, description):
        ax_ls = ax if isinstance(ax, list) else [ax]    # cosider multiple axis
        if fig_id in self.figures:
            print(f"Figure ID {fig_id} is already in the figure manager")
            return
        if self.has_figure(fig_to_check=fig):
            print(f"Figure is already in the figure manager")
            return
        self.figures[fig_id] = {'figure': fig, 'axis': ax_ls, 'description': description}

    def get_figure(self, fig_id):
        return self.figures.get(fig_id, {}).get('figure')

    def get_axis(self, fig_id):
        return self.figures.get(fig_id, {}).get('axis')

    def show_figure(self, fig_id):
        fig = self.get_figure(fig_id)
        if fig:
            plt.figure(fig.number)
            plt.show()

    def save_figure(self, fig_id, filename):
        fig = self.get_figure(fig_id)
        if fig:
            fig.savefig(filename)
        else:
            print(f"No figure with ID {fig_id}")

    def close_figure(self, fig_id):
        fig = self.get_figure(fig_id)
        if fig:
            plt.close(fig)
            del self.figures[fig_id]
        else:
            print(f"No figure with ID {fig_id}")

    def close_all(self):
        plt.close('all')
        self.figures = {}

