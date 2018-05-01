import numpy as np
import matplotlib.pyplot as plt

"""
This file contains the utility helper functions that is used in the other classes
for performing sie basic computation

Author: Jeet
Date Modified
"""
import h5py

def write__to_h5_file(filename, **data):
    """
    This function writes the model data to an h5 file
    :param filename: name of the file to write
    :param data: A variable argument containing th data
    :return:
    """
    hf = h5py.File(filename)
    for key, value in data.items():
        hf.create_dataset(key, data=value)
    hf.close()


def get_random_Q_values(shape):
    """
    This method generates random numbers of given shape
    :param shape: shape of the the array of numbers to generate
    :return:
    """
    np.random.rand(shape=shape)


def get_predicted_Q_values(learning_model, state):
    """
    This function predicts teh Q value using the learning model
    :param learning_model: Deep Q leanning model
    :param state: current state
    :return:
    """
    # This  Method predicts the Q values for all the actions at a given state
    # using  the learning model
    Q = learning_model.predict(state)  # 1e-30 to avoid 0 values for this state always
    return Q


def epsilon_greedy_action(QValues, epsilon=0.01):
    """
    This method implements the epsilon greedy approach which takes a random action with a probability epsilon
    :param QValues: Q values of all the action for this state
    :param epsilon: value for controlling randomness
    :return:
    """
    # This  Method to choose action from a state using the
    # estimated Q values in epsilon-greedy fashion
    max_Q = np.max(QValues)

    equal_max = np.where(QValues == max_Q)[1]
    action = np.random.choice(equal_max)

    if np.random.random() < epsilon:
        random_action = np.random.randint(QValues.shape[1])
        action = random_action
        max_Q = QValues[0, action]

    return max_Q, action


def visualize(agent, start, goal, path, terrain_limit):
    """
    This method makes the plot for visualizing the results such as trajectory followed by the robot and
    the Contour plot for Q values
    :param agent: robot
    :param start: start state
    :param goal: goal state
    :param path: path followed by the robot
    :param terrain_limit: limit for the environment
    :return:
    """

    # Plot the trajectory
    plt.figure(figsize=(20, 7))
    plt.subplot(1, 2, 1)
    path_x = path[:, 1]
    path_y = path[:, 0]
    plt.plot(path_y,  path_x, color='black')

    plt.xlim([-terrain_limit[0], terrain_limit[0]])
    plt.ylim([-terrain_limit[1], terrain_limit[1]])

    # Plot the max Q Contour
    xs = np.round(np.arange(-terrain_limit[0], terrain_limit[0], 0.01).reshape(-1, 1), 3)
    ys = np.round(np.arange(-terrain_limit[1], terrain_limit[1], 0.01).reshape(-1, 1), 3)
    X, Y = np.meshgrid(xs, ys)

    # state the state matrix
    states = np.zeros((xs.shape[0]*ys.shape[0], 2))
    for i in range(xs.shape[0]):
        for j in range(ys.shape[0]):
            states[i*ys.shape[0]+j, 0] = xs[i, 0]
            states[i*ys.shape[0]+j, 1] = ys[j, 0]
    Q_values = get_predicted_Q_values(agent.learning_model, states)
    maxQ = np.max(Q_values, axis=1)
    # create A 2d matrix for denoting q value for each state
    Q = np.zeros((xs.shape[0], ys.shape[0]))
    for i in range(xs.shape[0]):
        for j in range(ys.shape[0]):
            Q[i, j] = maxQ[i*ys.shape[0]+j]

    plt.subplot(1, 2, 2)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    cs = plt.contourf(X, Y, Q)
    plt.colorbar(cs)
    plt.text(goal[1], goal[0], 'G')
    plt.text(start[1], start[0], 'S')
    plt.ylabel("max Q")
    plt.savefig('trajectory_and_Q_contour.png', bbox_inches='tight')
    plt.show()


