import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from numba import cuda

import read_maze

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Describing the actions that the agent can take in the maze
# It can move horizontally or vertically or stop in the position
action_count = 5
action_def = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'STAY']
action_list = {'LEFT': torch.tensor([0, -1]),
           'RIGHT': torch.tensor([0, 1]),
           'UP': torch.tensor([-1, 0]),
           'DOWN': torch.tensor([1, 0]),
           'STAY': torch.tensor([0, 0])}

# Loading in the maze and reading the size of the rows and columns
# In the provided maze the size is 199 for both, however these lines would make it easier to load in a different maze if needed
read_maze.load_maze()
rows = read_maze.maze_cells.shape[0]
cols = read_maze.maze_cells.shape[1]

# Creating an empty Qtable instance
Qtable = torch.zeros((rows, cols, action_count))

# Creating a list that would record dead end maze cells so that they could be avoided in later episodes
dead_end_list = []

# gamma closer to one so that agent would rather consider future rewards.
gamma = 0.9


def explore_maze(fire_on = True, e=0.1):
    # Initially we made 0 steps
    steps = 0
    # Recording all the cells visited, at the start we are at coordinate [1,1]
    visited_cells = [(1, 1)]

    # Setting the position of the agent
    position = torch.tensor([1, 1])

    # Starting reward and initialising a boolean for reaching the end
    overall_reward = 0
    reached_end = False

    # Intersection stack
    intersections = []
    # Recording the actions taken at intersections so that dead ends could be identified
    intersection_actions = []

    # Instantiating a metric for recording possible moves of at a certain position
    sample_count = torch.zeros((rows, cols, action_count))

    # A while loop that continues till the end of the maze is reached 
    # It also stops when the reward becomes too low, as that indicates that the algorithm has likely made too many inacurate moves
    while overall_reward > -1500 and not reached_end:
        # instantiating x and y coordinates
        x = position[0]
        y = position[1]
    
        # Getting fire cells
        if fire_on:
            fire_is_on = np.argwhere(read_maze.maze_cells[:, :, 1] > 0)    
        
        
        # Calling get_local_maze_information() from the provided read_maze file to observe the maze around the agents position
        around = read_maze.get_local_maze_information(x, y)

        # Counting the walls around the agent to determine whether he has approached a dead end
        # If a maze position is in the dead_end_list that means it leads to a dead end and is counted as a wall
        walls = 0
        options = 0
        # If a wall is above
        if around[0, 1, 0] == 0 or (x-1, y) in dead_end_list: 
            walls = walls + 1
        else:
            options = options + 1
        # If a wall is below
        if around[2, 1, 0] == 0 or (x+1, y) in dead_end_list: 
            walls = walls + 1
        else:
            options = options + 1
        # If a wall is left
        if around[1, 0, 0] == 0 or (x, y-1) in dead_end_list: 
            walls = walls + 1
        else:
            options = options + 1
        # If a wall is right
        if around[1, 2, 0] == 0 or (x, y+1) in dead_end_list:
            walls = walls + 1
        else:
            options = options + 1

        
        # If the agent is at a dead end, the position is returned to the previous intersection and the dead end is blocked off
        if not (x == 1 and y == 1) and walls >= 3:
            position = intersections.pop()
            intersection_action_taken = intersection_actions.pop()

            # Position tensor is updated
            position = torch.tensor([position[0], position[1]])

            # Adding the cell that lead to a dead end to a dead_end_list
            new_dead_end = (position[0] + intersection_action_taken[0], position[1] + intersection_action_taken[1])
            dead_end_list.append(new_dead_end)
        else:
            # Randomly choosing an action if not a dead end
            random_num = random.uniform(0, 1)
            if random_num < e:
                train_action = random.randint(0, action_count - 1)
            else:
                # Determining the action that maximizes the Q function the most
                train_action = torch.argmax(Qtable[x, y, :])
            current_action_def = action_def[train_action]
            current_action = action_list[current_action_def]

            # If there are 3 or more options for the agent to move we classify the position as an intersection
            if (x == 1 and y == 1) or options >= 3:
                # If this intersection is already recorded in the stack later intersections are removed
                if (x, y) in intersections:
                    intersection_i = intersections.index((x, y))
                    intersections = intersections[0: intersection_i]
                    intersection_actions = intersection_actions[0: intersection_i]

                # Add the intersection position to the list
                intersections.append((x, y))
                intersection_actions.append(current_action)

            # Assigning a new agent position
            potential_new_position = position + current_action
            potential_new_x = potential_new_position[0]
            potential_new_y = potential_new_position[1]

            # Removing reward and staying in the same position if the selected action leads to fire, dead_end, wall or an already visited cell
            # Leads to fire
            if around[1 + current_action[0]][1 + current_action[1]][1] > 0:
                new_position = position 
                reward = -0.25
            # Leads to dead end   
            elif (potential_new_x, potential_new_y) in dead_end_list:
                new_position = position
                reward = -2
            # Leads to wall
            elif around[1 + current_action[0]][1 + current_action[1]][0] == 0: 
                new_position = position
                reward = -0.8
            # Leads to already visited cell
            elif (potential_new_x, potential_new_y) in visited_cells:
                new_position = position 
                reward = -0.6
            # If the agent moves to a new cell we remove a small amount of the reward as we still want to minimise moves.
            else:
                new_position = potential_new_position
                visited_cells.append((potential_new_x, potential_new_y))
                reward = -0.05 

            new_x = new_position[0]
            new_y = new_position[1]

            # If the maze exit is reached we add reward
            if new_x == rows - 2 and new_y == cols - 2:
                reached_end = True
                reward = 100

            # Adding training action and the position of that action to the sample count
            sample_count[x, y, train_action] += 1

            # Finding the further action that would minimise the Q function
            next_train_action = torch.argmax(Qtable[new_x, new_y, :])
            alpha = 1/sample_count[x, y, train_action]

            # Updating the Qtable with the calculated values
            Qtable[x, y, train_action] = Qtable[x, y, train_action] + alpha * (reward + gamma * Qtable[new_x, new_y, next_train_action] - Qtable[x, y, train_action])

            # Printing the moves to a txt file
            txt_moves(x,y,steps,current_action_def, overall_reward, around)

            # Update the current position
            position = new_position

            # Adding the reward and increasing the step number
            overall_reward = overall_reward + reward
            steps = steps + 1

    return overall_reward, position





# Function for printing out the moves and the cell around information
def txt_moves(x,y,steps,current_action_def,overall_reward, around):
    with open('moves.txt', 'a') as f:

            text = "Move number - " + str(steps) + ":\n"
            text += "Current Agent Location: [" + str(x.item()) + ", " + str(y.item()) + "] \n"
            text += "Action: " + current_action_def + ", \n"
            text += "Reward: " + str(overall_reward) + "\n"
            f.write(text)


            text = "Around cells         Fire information\n"
            f.write(text)
            for k in range(3):
                    text = "[" + str(around[k, 0, 0]) + "]" + "[" + str(around[k, 1, 0]) + "]" + "[" + str(around[k, 2, 0]) + "]                 "
                    text = text + "[" + str(around[k, 0, 1]) + "]" + "[" + str(around[k, 1, 1]) + "]" + "[" + str(around[k, 2, 1]) + "]\n"
                    f.write(text)
            f.write('-------------------------------------------------------------------\n')

    
def train(episodes=50):

    overall_rewards_list = []
    e=0.1
    decay = 0.99

    # Run training for set amount of episodes
    for epoch in range(episodes):
        episode_reward, end_pos = explore_maze(fire_on=False)
        overall_rewards_list.append(episode_reward)
        e = e * decay
        print("Epoch: ", epoch, "    Agent ends in: ", end_pos, " Total reward: ", episode_reward)


    # Plot total rewards for each epoch
    fig = plt.figure(2)
    ax = fig.gca()
    ax.plot(np.arange(1, episodes+1), overall_rewards_list ,'r')
    ax.set_title('Reward over episodes', fontsize=12)
    ax.set_xlabel('episode')
    ax.set_ylabel('Reward')
    ax.grid()
    fig.savefig('reward_change.png')

    # Save Q-table to file Qtable.pt
    torch.save(Qtable, 'Qtable.pt')


# IF YOU WANT TO RETRAIN THE AGENT BY GENERATING A NEW Q-TABLE UNCOMMENT THE TWO LINES BELOW (WARNING: LENGTHY)
#print("Training...")
#train(episodes=100)

# Load trained Q-table
Qtable = torch.load('Qtable.pt')

print("Using provided Q-table to find solution to the provided maze...")

# The fire maze is being solved with the provided Q-table file
explore_maze(fire_on=True)

print("Check moves.txt for the solution")