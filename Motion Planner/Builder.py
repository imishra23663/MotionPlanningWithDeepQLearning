from klampt import Geometry3D
import numpy as np
"""
This class represents the world in which the robot will be
used in to find a goal configuration from teh initial configuration

Author: Jeet
Date Modified: 04/11/2018

This world contains below components

"""

class Builder:
    def __init__(self, objects_dir):
        self.objects_dir = objects_dir
        self.objects = np.zeros((0, 3))

    def make_cube(self, world, width, depth, height, x, y, z, wall_thickness, color, name,
                  object_file="cube.off", object_type="terrain"):
        """
        This function created a cube object based on the given dimension and position
        :param world: world object
        :param width: width of the cube
        :param depth: depth of the cube
        :param height: height of the cube
        :param x: X coordinate of the  of the cube center
        :param y: Y coordinate of the  of the cube center
        :param z: Z coordinate of the  of the cube center
        :param wall_thickness: wall thickness of the cube
        :param object_file: file of the cube object
        :return:
        """
        object_path = self.objects_dir + object_file

        # create the Geometry for all the faces of the object
        left = Geometry3D()
        right = Geometry3D()
        front = Geometry3D()
        back = Geometry3D()
        bottom = Geometry3D()
        top = Geometry3D()

        left.loadFile(object_path)
        right.loadFile(object_path)
        front.loadFile(object_path)
        back.loadFile(object_path)
        bottom.loadFile(object_path)
        top.loadFile(object_path)

        left.transform([wall_thickness, 0, 0, 0, depth, 0, 0, 0, height], [-width*0.5, -depth*0.5, 0])
        right.transform([wall_thickness, 0, 0, 0, depth, 0, 0, 0, height], [width * 0.5, -depth * 0.5, 0])
        front.transform([-width, 0, 0, 0, wall_thickness, 0, 0, 0, height], [width * 0.5, -depth * 0.5, 0])
        back.transform([width, 0, 0, 0, wall_thickness, 0, 0, 0, height], [-width * 0.5, depth * 0.5, 0])
        bottom.transform([width, 0, 0, 0, depth, 0, 0, 0, wall_thickness], [-width * 0.5, -depth * 0.5, 0])
        top.transform([width, 0, 0, 0, depth, 0, 0, 0, wall_thickness],
                      [-width * 0.5, -depth * 0.5, height-wall_thickness])
        cubegeom = Geometry3D()

        cubegeom.setGroup()
        for i, elem in enumerate([left, right, front, back, bottom, top]):
            g = Geometry3D(elem)
            cubegeom.setElement(i, g)

        if object_type == "rigid" or object_type == "goal":
            cube = world.makeRigidObject(name)
        else:
            cube = world.makeTerrain(name)
        cube.geometry().set(cubegeom)
        cube.appearance().setColor(color[0], color[1], color[2], color[3])
        cube.geometry().translate((x, y, z))

        return cube

    def make_objects(self, world, n_objects, type, width, depth, height, thickness, terrain_limit, color, goal_pc=None):
        """
        This method creates given number of objects of the given type such as goal or obstacle
        :param world: robot's world
        :param n_objects: number of objects to create
        :param width: width of the object
        :param type:
        :param height: height of the object
        :param depth: wdepth of the object
        :param thickness: thickness of the walls of the object
        :param terrain_limit: 1d array representing the limit of the terrain
        :param color: color for the object
        :param goal_pc: coordinate of teh goal
        :return:
        """

        obstacles = []
        for i in range(n_objects):
            while True:

                # Generate random location for objects and check if they are colliding generate new position
                x = np.random.uniform(-terrain_limit[0]-width, terrain_limit[0]-width)
                y = np.random.uniform(-terrain_limit[1]-width, terrain_limit[1]-width)
                z = terrain_limit[2]    # will always be constant
                found = False
                for object_pc in self.objects:
                    if x == object_pc[0] and y == object_pc[1] and z == object_pc[2]:
                        found = True
                        break

                if not found:
                    break

            if type == "goal":
                if self.objects.shape[0] == 0:
                    self.objects = np.array([[x, y, z]])
                else:
                    self.objects = np.concatenate((self.objects, np.array([[x, y, z]])), axis=1)
                goal = self.make_cube(world, width, depth, height, goal_pc[0], goal_pc[1], goal_pc[2], thickness, color,
                                      name="goal", object_type=type)

                break   # Since Only one goal is allowed
            else:
                if self.objects.shape[0] == 0:
                    self.objects = np.array([[x, y, z]])
                else:
                    self.objects = np.concatenate((self.objects, np.array([[x, y, z]])), axis=1)

                obstacles.append(self.make_cube(world, width, depth, height, x, y, z, thickness, color,
                                                name="obstacle"+str(i), object_type=type))
