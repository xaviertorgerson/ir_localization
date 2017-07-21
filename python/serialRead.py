import serial, sys
from Tkinter import *
from PointBuffer import PointBuffer
import math
import numpy

# Define constants
canvas_width = 1022
canvas_height = 1022

bottom_left = None
bottom_right = None
top_left = None
top_right = None

# Initialize GUI
tk = Tk()
tk.title("ir localization")
canvas = Canvas(tk, width=canvas_width, height=canvas_height)
canvas.pack()

def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    """Return affine transform matrix to register two point sets.

    v0 and v1 are shape (ndims, \*) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.

    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean transformation matrix
    is returned.

    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.

    The returned matrix performs rotation, translation and uniform scaling
    (if specified).

    >>> v0 = [[0, 1031, 1031, 0], [0, 0, 1600, 1600]]
    >>> v1 = [[675, 826, 826, 677], [55, 52, 281, 277]]
    >>> affine_matrix_from_points(v0, v1)
    array([[   0.14549,    0.00062,  675.50008],
           [   0.00048,    0.14094,   53.24971],
           [   0.     ,    0.     ,    1.     ]])
    >>> T = translation_matrix(numpy.random.random(3)-0.5)
    >>> R = random_rotation_matrix(numpy.random.random(3))
    >>> S = scale_matrix(random.random())
    >>> M = concatenate_matrices(T, R, S)
    >>> v0 = (numpy.random.rand(4, 100) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = numpy.dot(M, v0)
    >>> v0[:3] += numpy.random.normal(0, 1e-8, 300).reshape(3, -1)
    >>> M = affine_matrix_from_points(v0[:3], v1[:3])
    >>> numpy.allclose(v1, numpy.dot(M, v0))
    True

    More examples in superimposition_matrix()

    """
    v0 = numpy.array(v0, dtype=numpy.float64, copy=True)
    v1 = numpy.array(v1, dtype=numpy.float64, copy=True)

    print(v0.shape[0])
    print(v0.shape[1])

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -numpy.mean(v0, axis=1)
    M0 = numpy.identity(ndims+1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -numpy.mean(v1, axis=1)
    M1 = numpy.identity(ndims+1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = numpy.concatenate((v0, v1), axis=0)
        u, s, vh = numpy.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims:2*ndims]
        t = numpy.dot(C, numpy.linalg.pinv(B))
        t = numpy.concatenate((t, numpy.zeros((ndims, 1))), axis=1)
        M = numpy.vstack((t, ((0.0,)*ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = numpy.linalg.svd(numpy.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = numpy.dot(u, vh)
        if numpy.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= numpy.outer(u[:, ndims-1], vh[ndims-1, :]*2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = numpy.identity(ndims+1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = numpy.sum(v0 * v1, axis=1)
        xy, yz, zx = numpy.sum(v0 * numpy.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = numpy.sum(v0 * numpy.roll(v1, -2, axis=0), axis=1)
        N = [[xx+yy+zz, 0.0,      0.0,      0.0],
             [yz-zy,    xx-yy-zz, 0.0,      0.0],
             [zx-xz,    xy+yx,    yy-xx-zz, 0.0],
             [xy-yx,    zx+xz,    yz+zy,    zz-xx-yy]]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = numpy.linalg.eigh(N)
        q = V[:, numpy.argmax(w)]
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(numpy.sum(v1) / numpy.sum(v0))

    # move centroids back
    M = numpy.dot(numpy.linalg.inv(M1), numpy.dot(M, M0))
    M /= M[ndims, ndims]
    return M

def draw_lines(current_point, last_point):
    if(current_point is not None and last_point is not None):
        (last_x, last_y) = last_point
        (current_x, current_y) = current_point
        canvas.create_line(last_x, last_y, current_x, current_y, fill="red")

# Initialize serial connection
ser = serial.Serial(sys.argv[1], 19200)
point_buffer = PointBuffer()

calibrate = True

print("Calibrate bottom left of screen")
while(True):
    line = ser.readline()
    point_buffer.update(line)


    if((bottom_left is None) or (bottom_right is None) or (top_left is None) or (top_right is None)):
        if(bottom_left is None):
            if(point_buffer.strongest_point[-1] is not None and calibrate):
                bottom_left = point_buffer.strongest_point[-1]
                calibrate = False
                print("Bottom left at point", point_buffer.strongest_point[-1])
                print("Calibrate bottom right of screen")
            elif(point_buffer.strongest_point[-1] is None):
                calibrate = True
        elif(bottom_right is None):
            if(point_buffer.strongest_point[-1] is not None and calibrate):
                bottom_right = point_buffer.strongest_point[-1]
                calibrate = False
                print("Bottom right at point", point_buffer.strongest_point[-1])
                print("Calibrate top left of screen")
            elif(point_buffer.strongest_point[-1] is None):
                calibrate = True
        elif(top_left is None):
            if(point_buffer.strongest_point[-1] is not None and calibrate):
                top_left = point_buffer.strongest_point[-1]
                calibrate = False
                print("Top left at point", point_buffer.strongest_point[-1])
                print("Calibrate top right of screen")
            elif(point_buffer.strongest_point[-1] is None):
                calibrate = True
        elif(top_right is None):
            if(point_buffer.strongest_point[-1] is not None and calibrate):
                top_right = point_buffer.strongest_point[-1]
                calibrate = False
                print("Top right at point", point_buffer.strongest_point[-1])
                M = affine_matrix_from_points([(bottom_left[0],bottom_left[1],0),(bottom_right[0], bottom_right[1],0), (top_right[0], top_right[1], 0)], [(0,0,0),(canvas_width,0,0),(canvas_width,canvas_height,0)], shear=False, scale=True, usesvd=True)
                #point_buffer.transformation_matrix = M
                print(M)
            elif(point_buffer.strongest_point[-1] is None):
                calibrate = True
    else:
        draw_lines(point_buffer.strongest_point[-1], point_buffer.strongest_point[-2]) # If points changed draw a new line
        #draw_lines(point_buffer.transformation_matrix[-1], point_buffer.transformation_matrix[-2]) # If points changed draw a new line

    # Update GUI
    tk.update_idletasks()
    tk.update()

ser.close()
