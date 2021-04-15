import cv2
import numpy
import random
import math


class Particle:
    def __init__(self, dot, color, vector):
        self.color = color
        self.vector = vector
        up = (dot[0] + int(self.vector[0]*random.random()), dot[1] + int(self.vector[1]*random.random()))
        down = (dot[0] - int(self.vector[0]*random.random()), dot[1] - int(self.vector[1]*random.random()))
        self.polygon = [dot, up, (dot[0] + self.vector[0], dot[1] + self.vector[1]), down]
        self.delete = False

    def draw(self, frame):
        cv2.fillPoly(frame, [numpy.array(self.polygon)], self.color)

    def update(self):
        for index, dot in enumerate(self.polygon):
            self.polygon[index] = (dot[0] + self.vector[0], dot[1], self.vector[1])
        dif02 = self.polygon[0][0] - self.polygon[2][0], self.polygon[0][1] - self.polygon[2][1]
        dif13 = self.polygon[1][0] - self.polygon[3][0], self.polygon[1][1] - self.polygon[3][1]
        self.polygon[0] = self.polygon[0][0] - dif02[0] // 4, self.polygon[0][1] - dif02[1] // 4
        self.polygon[2] = self.polygon[2][0] + dif02[0] // 4, self.polygon[0][1] + dif02[1] // 4
        self.polygon[1] = self.polygon[1][0] - dif13[0] // 4, self.polygon[1][1] - dif13[1] // 4
        self.polygon[3] = self.polygon[3][0] + dif13[0] // 4, self.polygon[3][1] + dif13[1] // 4
        if self.polygon[0] == self.polygon[2] or self.polygon[1] == self.polygon[3]:
            self.delete = True
