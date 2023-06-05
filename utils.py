from __future__ import annotations
from operator import truediv
from typing import List

class Point():
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class PointEncoded(Point):
    """sistema de coordenadas absoluto"""
    def __init__(self, x: float, y: float):
        super().__init__(x, y)
        self.is_start_of_line = 0
        self.is_end_of_line = 0
    
    def set_start_of_line(self):
        self.is_start_of_line = 1

    def set_end_of_line(self):
        self.is_end_of_line = 1
    
    def format(self):
        return "({}, {}), ".format(self.x, self.y)
    
class Line():
    def __init__(self, points: List[PointEncoded]):
        self.points = points
        self.start_point = self._get_start_point(points)
        self.end_point = self._get_end_point(points)

    def add_point(self, point: PointEncoded):
        self.points += [point]
        return True
   
    def _get_start_point(self, points: List[PointEncoded]):
        for point in points:
            if point.is_start_of_line:
                return point

    def _get_end_point(self, points: List[PointEncoded]):
        for point in points:
            if point.is_end_of_line:
                return point

    def format(self):
        line = ""
        for point in self.points:
            line += point.format()
        return line

class Drawing():
    def __init__(self, lines: List[Line]):
        self.lines = lines

    def union(self, drawing: Drawing):
        self.lines += drawing.lines

    def pretty_printer(self):
        print("printing drawing...")
        for i, line in enumerate(self.lines):
            print("Line {}: {}".format(i+1, line.format()))
            