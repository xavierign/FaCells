from __future__ import annotations
from operator import truediv
from typing import List
import statistics

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
        self.len_longest_line = None
        self.len_shortest_line = None
        self.mean_len_lines = None
        self.median_len_lines = None
        self.mode_len_lines = None

    def union(self, drawing: Drawing):
        self.lines += drawing.lines

    def get_x_values(self):
        x_values = []
        for line in self.lines:
            
            x_values += line.x


    def pretty_printer(self):
        print("Printing drawing...")
        for i, line in enumerate(self.lines):
            print(f"Line {i+1}: {line.format()}")

        print(f"Number of lines in drawing: {len(self.lines)}")

        self.get_stats()
        print("\nLines stats:")
        print(f"Longest line: {self.len_longest_line}")
        print(f"Shortest line: {self.len_shortest_line}")
        print(f"Mean length lines: {self.mean_len_lines:.2f}")
        print(f"Median length lines: {self.median_len_lines}")
        print(f"Mode length lines: {self.mode_len_lines}")
        
    def get_stats(self):
        """Populates length of longest and shortest line, mean, mode and median"""
        self.len_longest_line = 0
        self.len_shortest_line = len(self.lines[0].points)
        len_lines = []
        for i, line in enumerate(self.lines):
            length_line = len(line.points)
            len_lines += [length_line]
            if length_line < self.len_shortest_line:
                self.len_shortest_line = length_line
            if length_line > self.len_longest_line:
                self.len_longest_line = length_line

        self.mean_len_lines = statistics.mean(len_lines)
        self.median_len_lines = statistics.median(len_lines)
        self.mode_len_lines = statistics.mode(len_lines)
        
        
            