import serial, sys
from Tkinter import *
import math
import numpy

class PointBuffer:

  def __init__(self):
    self.all_points = []
    self.strongest_point = []

    self.aspect_ratio = 1;
    self.drawable_point = []

    self.transformation_matrix = None
    self.transformed_point = []


  def update(self, line):
    self.add(self.decode(line))
    self.display()

    '''
    # Print new points
    if(new_point is not None):
      print("\n")
      if(len(new_points) > 1):
        print("All points", new_points)
      (x, y) = new_point
      print("Strongest point @", x, y)
      if(self.transformed_point is not None):
        if(len(self.transformed_point) > 0):
          (transformed_x, transformed_y) = self.transformed_point[-1]
          print("Transformed point @", transformed_x, transformed_y)
    '''

  def display(self):
    if(self.all_points[-1] is not None):
      print("All points",self.all_points[-1])
      print("Main point",self.strongest_point[-1])
      print("GUI point",self.drawable_point[-1])
      if(len(self.transformed_point) > 0):
        print("Trans point",self.transformed_point[-1])
      print("\n")


  def add(self, new_points):
    self.all_points.append(new_points)
    self.strongest_point.append(self.get_strongest_point(new_points))
    self.drawable_point.append(self.get_drawable_point(self.strongest_point[-1]))
    if(self.transformation_matrix is not None):
        self.transformed_point.append(self.get_transformed_point(self.drawable_point[-1]))

  def decode(self, line):
    output = line.strip('\n').strip('\r').split(',')

    if (len(output) == 8): # Validate string

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

  def get_strongest_point(self, new_points):
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

  def get_drawable_point(self, new_point):
    if (new_point is None):
      return None

    (x, y) = new_point
    return (1022-x, y)

  def get_transformed_point(self, new_point):
    if (new_point is None):
      return None

    #v0 = numpy.array([[new_point[0]],[new_point[1]], [0], [1]])
    v0 = numpy.array([[new_point[0]],[new_point[1]], [1]])
    transformed_vector = self.transformation_matrix*v0
    return (transformed_vector.item(0), transformed_vector.item(1))

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
