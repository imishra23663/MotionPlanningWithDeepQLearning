import sys
import time
import numpy as np
from kinematics.sphero6DoF import sphero6DoF
from klampt.model import ik,coordinates
from buildWorld import *
from klampt import vis
from klampt.math import so3
import klampt.model.collide as collide
from functions import *
import copy

from Builder import Builder


class Simulation:
    def __init__(self, world):
        self.world = world
        self.robot = None
        self.collision_checker = None
        self.step_size = 0.05
        self.actions = np.radians(np.arange(180, -180, -30))
        # self.actions = np.array([np.pi, np.pi/2, 0, -np.pi/2])
        # self.actions = np.array([-np.pi, 0, np.pi / 2, -np.pi / 2])
        self.reward_system = {"collision": -50, "boundary": -50, "free": -5, "goal": 1000}
        self.distance_factor = 0.02
        self.start_pc = None
        self.goal_pc = None
        self.goal_range = 0.11  # size of the goal
        self.terrain_limit = [1, 1, 0]
        self.vis = None

    def reset(self, spc):
        # This function resets the robot to the start position again
        self.robot.setConfig(spc)

    def check_collision(self):
        is_goal = False
        for iT in range(self.world.numTerrains()):
            collRT0 = self.collision_checker.robotTerrainCollisions(self.world.robot(0), iT)
            collision_flag = False
            for i, j in collRT0:
                collision_flag = True
                strng = "Robot collides with " + j.getName()
                vis.addText("textCol", strng)
                vis.setColor("textCol", 1, 0, 0)
                break

        for iR in range(self.world.numRigidObjects()):
            collRT2 = self.collision_checker.robotObjectCollisions(self.world.robot(0), iR)
            for i, j in collRT2:
                if j.getName() != "goal":
                    collision_flag = True
                    strng = self.world.robot(0).getName() + " collides with " + j.getName()
                    vis.addText("textCol", strng)
                    vis.setColor("textCol", 1, 0, 0)
                else:
                    is_goal = False

        if collision_flag:
            vis.addText("textCol", "Collision")
            vis.setColor("textCol", 1, 0.0, 0.0)

        if not collision_flag:
            vis.addText("textCol", "No collision")
            vis.setColor("textCol", 0.4660, 0.6740, 0.1880)

        return collision_flag, is_goal

    def check_goal_distance(self, cpc):
        """
        This method checks the distance of the gaol from the current position of thr robot
        :param cpc: current position coordinate of the robot
        :return:
        """

        if type(cpc) is np.ndarray:
            distance = np.sqrt(np.square(cpc[0, 0] - self.goal_pc[0]) + np.square(cpc[0, 1] - self.goal_pc[1]))
            if distance < self.goal_range:
                return 0
            else:
                # return the distance from the goal
                return distance
        else:
            distance = np.sqrt(np.square(cpc[0] - self.goal_pc[0]) + np.square(cpc[1] - self.goal_pc[1]))
            if distance < self.goal_range:
                return 0
            else:
                # return the distance from the goal
                return distance

    def create(self, start_pc, goal_pc):
        """
        This method cretes the simulation
        :param start_pc: robot's initial position coordinate
        :param goal_pc: goal position coordinate
        :return:
        """
        print "Creating the Simulator"
        object_dir = "./"
        self.start_pc = start_pc
        self.goal_pc = goal_pc
        coordinates.setWorldModel(self.world)
        getDoubleRoomDoor(self.world, 8, 8, 1)

        builder = Builder(object_dir)
        # Create a goal cube
        n_objects = 1
        width = 0.1
        depth = 0.1
        height = 0.1
        x = goal_pc[0]
        y = goal_pc[1]
        z = goal_pc[2]/2
        thickness = 0.005
        color = (0.2, 0.6, 0.3, 1.0)

        builder.make_objects(self.world, n_objects, "goal", width, depth, height, thickness, self.terrain_limit,
                             color, self.goal_pc)

        # Create a obstacle cube
        n_objects = 4
        width = 0.2
        depth = 0.2
        height = 0.2
        thickness = 0.001
        color = (0.8, 0.2, 0.2, 1.0)
        builder.make_objects(self.world, n_objects, "rigid", width, depth, height, thickness, self.terrain_limit,
                             color)

        self.vis = vis
        vis.add("world", self.world)
        # Create the view port
        vp = vis.getViewport()
        vp.w, vp.h = 800, 600
        vp.x, vp.y = 0, 0
        vis.setViewport(vp)

        # Create the robot
        self.robot = sphero6DoF(self.world.robot(0), "", None)
        self.robot.setConfig(start_pc)

        # Create the axis representation
        # vis.add("WCS", [so3.identity(), [0, 0, 0]])
        # vis.setAttribute("WCS", "size", 24)

        # Add text messages component for collision check and robot position
        vis.addText("textCol", "No collision")
        vis.setAttribute("textCol", "size", 24)

        vis.addText("textStep", "Steps: ")
        vis.setAttribute("textStep", "size", 24)
        self.collision_checker = collide.WorldCollider(self.world)

        vis.setWindowTitle("Simulator")
        vis.show()

        return vis, self.robot, self.world

    def get_current_state(self):
        """
        This method returns teh current configuration of the robot
        :return:
        """

        config = self.robot.getConfig()
        state = np.array([[config[0], config[1]]])
        return state

    def get_next_state(self, action, step_count, query=False):
        """
        This method implements the take action functionality by asking the robot to take the action, and return the
        next robot configuration
        :param action: action to take
        :param step_count: Number of steps taken by the robot so far
        :param query: A boolean variable denotes weather it slearning phase or query phase, if it is a query phase
                      add a delay for better visualization
        :return:
        """
        pc = self.robot.getConfig()
        old_pc = copy.deepcopy(pc)
        # Compute the new positions
        pc[0] = pc[0] + self.step_size * np.cos(self.actions[action])
        pc[1] = pc[1] + self.step_size * np.sin(self.actions[action])
        pc[2] = pc[2]
        pc[3] = 0
        pc[4] = 0
        pc[5] = 0

        self.robot.setConfig(pc)
        boundary_reached = False
        if np.abs(pc[0]) > self.terrain_limit[0] or np.abs(pc[1]) > self.terrain_limit[1]:
            boundary_reached = True

        q2f = ['{0:.2f}'.format(elem) for elem in pc]
        label = "Steps: " + str(step_count)
        vis.addText("textStep", label)
        cpc = self.robot.getConfig()

        # compute the distance
        distance = self.check_goal_distance(cpc)

        collision, is_goal = self.check_collision()
        goal_reached = False

        # Update the robot positions
        if boundary_reached:
            self.robot.setConfig(old_pc)
            reward = self.reward_system['boundary']
            # print("Hit Boundary", reward)
            pc = old_pc
        elif collision:
            self.robot.setConfig(old_pc)
            reward = self.reward_system['collision']
            # print("Collision", reward)
            # Don't let it cross the boundary
            pc = old_pc
        elif distance==0:
            reward = self.reward_system['goal']
            goal_reached = True
        else:
            reward = self.reward_system['free']

        # send only the x and y position
        state = np.array([[pc[0], pc[1]]])
        time.sleep(0.05)
        if query:
            time.sleep(0)
        return reward, state, goal_reached



