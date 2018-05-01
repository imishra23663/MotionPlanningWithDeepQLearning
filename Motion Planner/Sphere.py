"""
This class represents a sphere object
"""
from shapely.geometry import Point


class Sphere:
    def __init__(self, x, y, z, radius):
        self.pc = [x, y, z]
        self.radius = radius
        self.object = Point(x, y, z).buffer(radius)

    def setConfig(self, pc):
        self.object = Point(pc[0], pc[1], pc[2]).buffer(self.radius)

    def getConfig(self):
        return self.pc


