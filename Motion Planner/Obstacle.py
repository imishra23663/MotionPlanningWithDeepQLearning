"""
This class represents and item having x, y and ,z coordinate of its center and
the length, width and height
"""
from shapely.geometry.polygon import Polygon


class Obstacle:
    def __init__(self, x, y, z, l, w, h):
        self.x = x
        self.y = y
        self.z = z
        self.l = l
        self.w = w
        self.h = h
        self.type = type
        self.object = Polygon([(x, y, z), (x, y + w, z), (x + l, y + w, z), (x + l, y + w, z),
                               (x, y, z + h), (x, y + w, z + h), (x + l, y + w, z + h), (x + l, y, z + h)])

    def check_collion(self, object_item):
        """
        This function check if the agent collides with the polygon
        :param object_item:
        :return:
        """
        #print([self.object, object_item.object])
        return self.object.intersects(object_item.object)
