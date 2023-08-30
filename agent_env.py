import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

# continous state, continuous action, 2d actions. a goal
# punish touch the wall, reward when hit the goal
# state type ndarray(x,y), action type ndarray(x,y)

class ContinuousMap2D(gym.Env):
    def __init__(self, map_size=10):
        # max_history: maximum number of history states stored
        super(ContinuousMap2D, self).__init__()
        self.x = 0
        self.y = 0
        self.map_size = map_size # map_size of the grid map
        self.target_x, self.target_y = 0, 0
        self.action_space = spaces.Box(low=np.array([0, 0]),
                                       high=np.array([1.5, 1.5]),
                                       dtype=np.float32)   # actions x and y
        self.observation_space = spaces.Box(low=np.array([0, 0]),
                                            high=np.array([self.map_size, self.map_size]),
                                            dtype=np.float32)   # observe coordinate x and y
        self.state = None
        self.trajectory = []
        self.goal = np.array([5, 5])
        # Setting up the plot
        self.fig, self.ax = plt.subplots()
        # plt.ion()
        self.ax.grid(which='both')

    def step(self, action):
        # Take a copy of the current position to check for boundary crossing later
        old_pos = self.state.copy()

        # Take action
        potential_next_state = self.state**2 + action
        self.state = potential_next_state if self.observation_space.contains(potential_next_state) else self.state
        observation = self.state.copy()

        if np.array_equal(old_pos, self.state) <= 0:
            reward = -5  # Larger negative reward for going out of bounds
        elif sum(abs(old_pos - self.state)) <= 0.1:
            reward = -1  # Regular step penalty for move too small

        # # direction guide, kind of cheating
        # if sum(abs(self.goal - self.state)) < sum(abs(self.goal - old_pos)):
        #     reward = 5
        # else:
        #     reward = -1

        # reward if hit the target
        if sum(abs(self.goal - self.state)) <= 0.1:
            reward = 20
            done = True
        else:
            done = False
        info = {}

        # self.trajectory.append(self.state.copy())

        return observation, reward, done, info

    def reset(self):
        # Set robot position to a random location within the grid
        # np.random.seed(28)
        self.state = self.observation_space.sample()
        self.clear_trajectory()
        return self.state.copy()

    def render(self, mode='training'):
        # Clear the previous trajectory points
        self.ax.clear()

        # Setting grid spacing
        self.ax.set_xticks(np.arange(0, self.map_size, 1))
        self.ax.set_yticks(np.arange(0, self.map_size, 1))

        # plt.ion()  # Turn on interactive mode
        # Redraw grid
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.grid(which='both')

        # Plot the trajectory
        if self.trajectory:
            traj_x, traj_y = zip(*self.trajectory)
            self.ax.plot(traj_x, traj_y, "ro", label="Trajectory")

        # Plot current robot position
        self.ax.plot(self.state[0], self.state[1], "ro", label="Robot")

        # Show the updated plot
        plt.draw()
        plt.pause(0.1)  # Pause a bit so that the plot gets updated
        pass

    def clear_trajectory(self):
        self.trajectory = []
        # self.render()  # Update the plot after clearing

    def close(self):
        return None