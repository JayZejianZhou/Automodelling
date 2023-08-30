# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from agent_env import *
from utils import *
from DTAgent import *

file_loc = "models/"
file_prefix = "test1_"
run_log_dir = "run2"
mode = 0    # run mode, 0 - train, 1 - test once, 2 - test all saved models
test_model = 550 # the index of the saved model being tested in mode 1
save_interval = 50 # save episode duration
num_episodes = 600
tensor_board_interval = 10 # update tensor board interval
flag_tensor_board = False

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = ContinuousMap2D() # a customized gym to create synthetic data
    total_rewards = np.zeros(num_episodes)
    check_GPU_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # enable GPU
    # device = 'cpu'
    agent = DTAgent(env, device=device)
    figs_manager = FigureManager()
    figs_manager.register_figure(fig_id="state_render", fig=env.fig, ax=env.ax,
                                 description="This is the environment rendering figure")
    if flag_tensor_board:
        writer = SummaryWriter(log_dir="runs/" + run_log_dir)
    # Add model graph to TensorBoard
    # writer.add_graph(agent.q_function, input_to_model=torch.rand(3,))
    else:
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            total_reward, ep_loss = 0, 0  # Track total reward and NN's loss for this episode
            steps = 0  # keep track of the steps
            flag_render = False  # render every save points

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.memory.push(torch.tensor(state).to(device),
                                  torch.tensor(action).to(device),
                                  torch.tensor(next_state).to(device),
                                  torch.tensor([reward]).to(device),
                                  done)  # store the transition in memory once not done
                # perform one step of optimization on the cost function
                loss = agent.learn(episode=episode)
                ep_loss += loss if len(agent.memory) > agent.batch_size else 0
                state = next_state
                total_reward += reward
                steps += 1
                if steps > 200: break
            total_rewards[episode] = total_reward
            # Print episode number and its total reward
            print(
                f"Episode {episode + 1}/{num_episodes} - Episode Reward: {total_reward} - Steps: {steps} - Episode Loss: {ep_loss}")
            # Save the model
            if episode % save_interval == 0 and episode >= 1:
                save_model(agent.dt, name=file_loc + file_prefix + str(episode), save_mode=1)