import sys
import time
import numpy as np
from Robot import Robot
from klampt import WorldModel
from Simmulation import Simulation
from functions import visualize

if len(sys.argv)<=1:
    print "USAGE: kinematicSim.py [world_file]"
    exit()

world = WorldModel()
world_file = sys.argv[1]
res = world.readFile(world_file)
if not res:
    raise RuntimeError("Unable to load model " + world_file)

# np.random.seed(11)
np.random.seed(1711)
simulation = Simulation(world)
start_pc = [-0, 0, 0.05, 0, 0, 0]
goal_pc = [0.7, -0.5, 0.05, 0, 0, 0]
simulation.create(start_pc, goal_pc)
gamma = 0.99
time.sleep(5    )
agent = Robot(simulation, gamma)

if len(sys.argv) > 2 and sys.argv[2] == 'train':
    epochs = 600
    decay = 0.99
    max_step = 3000
    epsilon = 1
    epsilon_threshold = 0.001
    verbose = True
    verbose_iteration = 1
    agent.set_run_args(epochs, decay, epsilon_threshold, max_step, epsilon, start_pc, goal_pc, verbose,
                       verbose_iteration)
    agent.learn()
    # Save the model
    agent.learning_model.save("model/model.h5")
    # To suppress scientific notation
    np.set_printoptions(suppress=True)
else:
    agent.learning_model.load("model/model.h5")
    step, reward, trajectory = agent.get_path(start_pc)
    np.set_printoptions(suppress=True)
    visualize(agent, start_pc, goal_pc, trajectory, simulation.terrain_limit)

simulation.vis.kill()
