#Description: The code is a step-wise implementation of Q-Learning which is the fundamental concept of Reinforcement Learning. The objective is to make the agent learn how to solve a maze by itself.

#Import libraries
import numpy as np
import random
import matplotlib.pyplot as plt


# Set up important variables
actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
current_location = (0, 0)
end_location = (7, 7)
maze = [[2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0]]


# Initialize values
random_factor = 0.25
alpha = 0.1
rows = len(maze)
cols = len(maze[0])
rewards = np.full((rows,cols),-1)
rewards[end_location] = 0


#Construct Allowed states
allowed_states={}
key = 0
for i in range(0, rows):
    for j in range(0, cols):
        
        allowed_states[key]={'state':(i,j)}
        
        if (i!=0):
            if(maze[i-1][j] != 1):
                allowed_states[key].update({'U': np.random.uniform(-1, -0.1)})
        if (i!=rows-1):
            if(maze[i+1][j] != 1):
                allowed_states[key].update({'D': np.random.uniform(-1, -0.1)})
        if (j!=0):
            if(maze[i][j-1] != 1):
                allowed_states[key].update({'L': np.random.uniform(-1, -0.1)})
        if (j!=cols-1):
            if(maze[i][j+1] != 1):
                allowed_states[key].update({'R': np.random.uniform(-1, -0.1)})
        if(maze[i][j] == 1):                                                  
            del allowed_states[key]
         
        key = key + 1
        
        
##Helper Functions##

#Print maze
def print_maze(maze):
    print("█", end='')
    for col in maze[0]:
        print("█", end='')
    print("█")
    for row in maze:
        print("█", end='')
        for col in row:
            if (col == 0):
                print(' ', end='')
            elif (col == 1):
                print('█', end ='')
            elif (col == 2):
                print('O', end='')
        print("█")
    print("█", end='')
    for col in maze[0]:
        print("█", end='')
    print("█")
    
#Get key for dictionary
def get_key(dictionary, location):
    for key in dictionary:
        if dictionary[key]['state'] == location:
            return(key)
        
#Get new location
def get_next_location(next_move):
    if next_move == 'U':
        new_location = tuple([sum(val) for val in zip(actions['U'], current_location)])
    elif next_move == 'D':
        new_location = tuple([sum(val) for val in zip(actions['D'], current_location)])
    elif next_move == 'L':
        new_location = tuple([sum(val) for val in zip(actions['L'], current_location)])
    elif next_move == 'R':
        new_location = tuple([sum(val) for val in zip(actions['R'], current_location)])
    
    return(new_location)

#Get reward
def get_reward(state):
    return rewards[state[0], state[1]]

#Get the next move and next location
#Exploration(<factor) and Exploitation(>factor)
def get_next_move(allowed_states, random_factor, current_location):
    if (np.random.random() < random_factor):
        my_key = get_key(allowed_states, current_location)
        next_move = random.choice((list(allowed_states[my_key]))[1:])
    
    else:
        max_reward = -1e12
        my_key = get_key(allowed_states, current_location)
        temp=(list(allowed_states[my_key].items()))[1:]
        temp2 = dict(temp)
        next_move = max(temp2, key=temp2.get)
        current_max = temp2[max(temp2, key=temp2.get)]
        if (max_reward < current_max):
            max_reward = current_max
    
    return(next_move)

#Learning-update parameters
def get_state_history(allowed_moves, states):
    
    state_hist = []
    rewards_updated = [1]*(allowed_moves-1)
    i = allowed_moves - 2
    for i in range(i, 0, -1):                          
        if (i == allowed_moves - 2):                   #final location reward
            current_state = states[i]
            current_reward = get_reward(current_state)
            rewards_updated[i] = current_reward
            
        else:                         
            current_state = states[i]
            current_reward = get_reward(current_state)
            temp3 = current_reward + rewards_updated[i+1]
            rewards_updated[i] = temp3
        
            if (i == 1):                              #initial location reward
                i = i - 1
                current_state = states[i]
                current_reward = get_reward(current_state)
                temp3 = current_reward + rewards_updated[i+1]
                rewards_updated[i] = temp3

    state_hist = list(zip(states, rewards_updated))
    return(state_hist)

#Update Q table using state history
def update_q(allowed_moves, state_hist):
    current_location = (0,0)
    current_exp = []

    for i in range(0, allowed_moves-2):
        current_loc = state_hist[i][0]
        my_key = get_key(allowed_states, current_loc)
        current_exp_old = allowed_states[my_key][moves[i]]
        target_reward = state_hist[i+1][1]
        current_exp = current_exp_old + alpha*(target_reward - current_exp_old)
        allowed_states[my_key][moves[i]] = current_exp
    
    return(allowed_states)


# The main loop where you navigate to the end
attempt = 1
total_moves = []

while (attempt <= 5000): 

    allowed_moves = 1
    states = []
    moves = []
    current_location = (0,0)

    while (current_location != end_location and allowed_moves <= 1000):
        states.append(current_location)
        next_move = get_next_move(allowed_states, random_factor, current_location)
        moves.append(next_move)
        next_location = get_next_location(next_move)
        current_location = next_location
        allowed_moves += 1
    
        if (next_location == (7,7)):   #store last state and increment attempt to generalize indexing
            states.append(current_location)
            allowed_moves += 1
            break
    
    print('Attempt: ', attempt)
    print('Number of moves to reach the end = ', allowed_moves - 2)
    state_hist = get_state_history(allowed_moves, states)
    allowed_states = update_q(allowed_moves, state_hist)
    random_factor = random_factor - (1e-4*random_factor)
    
    total_moves.append(allowed_moves - 2)
    attempt += 1
    
    #Visualization - Heatmap
    if (attempt % 250 == 0):
        plt.figure()
        heat_arr = np.zeros((rows,cols))
        for i in range(0,len(state_hist)):
            temp_key = get_key(allowed_states,state_hist[i][0])
            if any(x == temp_key for x in list(allowed_states.keys())):
                location = state_hist[i][0]
                heat_arr[location[0]][location[1]] = heat_arr[location[0]][location[1]] + 1
        plt.imshow(heat_arr, cmap='hot', interpolation='nearest', vmin = 0, vmax = 5)
        plt.colorbar()
        plt.title('Attempt %i' %attempt)
        plt.show()
    
print('Training completed!')


#Visualization - Number of moves per iteration
attempts = list(range(1,5001))
plt.plot(attempts, total_moves)
plt.title('Number of moves per iteration')
plt.xlabel('Iterations')
plt.ylabel('Number of moves')
plt.show()

#Visualization - Print shortest path
for i in range(0, len(state_hist)):
    loc = state_hist[i][0]
    maze[loc[0]][loc[1]] = 2

print_maze(maze)
print("Congratulations! You made it to the end of the maze!")

