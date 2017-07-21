import serial, sys
from Tkinter import *
import math
import numpy

class PointBuffer:

  def __init__(self):
    self.all_points = []
    self.strongest_point = []
    self.transformed_point = []
    self.transformation_matrix

  def update(self, line):
    self.add(self.decode(line))
    new_points = self.all_points[-1]
    new_point = self.strongest_point[-1]

    # Print new points
    if(new_point is not None):
      if(len(new_points) > 1):
        print("Points seen", new_points)
      (x, y) = new_point
      print("Tracing point @", x, y)

  def add(self, new_points):
    self.all_points.append(new_points)
    self.strongest_point.append(self.find_strongest_point(new_points))
    #if(new_points is not None):
    #    v0 = numpy.array((self.strongest_point[-1][0],self.strongest_point[-1][1], 0))
    #    self.transformed_point.append(v0*self.transformation_matrix)
    #    print("Transformed Point=",self.transformed_point[-1])

  def decode(self, line):
    output = line.strip('\n').strip('\r').split(',')

    if (len(output) == 8): # Validate strin

      new_points = []
      if(int(output[0]) != 1023 or int(output[1]) != 1023):
        new_points.append((int(output[0]), int(output[1])))
      if(int(output[2]) != 1023 or int(output[3]) != 1023):
        new_points.append((int(output[2]), int(output[3])))
      if(int(output[4]) != 1023 or int(output[5]) != 1023):
        new_points.append((int(output[4]), int(output[5])))
      if(int(output[6]) != 1023 or int(output[7]) != 1023):
        new_points.append((int(output[6]), int(output[7])))

      if(len(new_points) != 0):
        return new_points

    return None

  def find_strongest_point(self, new_points):
    if (new_points is None):
      return None

    if(len(new_points) == 1):
      return new_points[0]

    last_valid_point = self.get_last_valid_point()
    if (last_valid_point is not None):
        (last_x, last_y) = last_valid_point
        lowest_displacement = 1023*1023
        lowest_idx = 0

        for (idx, (x, y)) in enumerate(new_points):
          displacement = math.sqrt(((x-last_x)*(x-last_x))+((y-last_y)*(y-last_y)))
          if (displacement < lowest_displacement):
            lowest_displacement = displacement
            lowest_idx = idx

        return new_points[lowest_idx]

    else:
        return new_points[0]

  def get_last_valid_point(self):
    for point in reversed(self.strongest_point):
      if(point is not None):
        return point
    return None

  #def get_last_character():
    #for i in range(self.length-1, 0, -1):
'''

# Initialize serial connection
ser = serial.Serial(sys.argv[1], 19200)
print(ser.name)

point_buffer = PointBuffer()
while(True):
    line = ser.readline()
    point_buffer.update(line)
'''
