#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time     : 2023/3/10
# @Author   : Hanwei Wang
# @Email    : wanghanweibnds2015@gmail.com
# @File     : test4mesh.py

import numpy as np

from mesh import Mesh

# data preparation
my_array = np.linspace(1, 36, 36).astype(int)

grid_2d = my_array.reshape([-1, 2])
grid_3d = my_array.reshape([-1, 3])
grid_4d = my_array.reshape([-1, 4])

grid_add_2d = np.zeros([3, 2])
grid_add_3d = np.zeros([3, 3])

mesh2d = Mesh()
mesh3d = Mesh(3)

# correct execute
mesh2d.set_grid(grid_2d)
mesh2d.add_grid(grid_add_2d)
# print(mesh2d.grid)
mesh3d.set_grid(grid_3d)
mesh3d.add_grid(grid_add_3d)
# print(mesh3d.grid)


# exception feedback
try:
    mesh2d.add_grid(grid_add_3d)
except IndexError as error:
    print("successfully detected!")
try:
    mesh2d.set_grid(grid_3d)
except IndexError as error:
    print("successfully detected!")

try:
    mesh3d.add_grid(grid_add_2d)
except IndexError as error:
    print("successfully detected!")
try:
    mesh3d.set_grid(grid_2d)
except IndexError as error:
    print("successfully detected!")
try:
    mesh3d.set_grid(grid_4d)
except IndexError as error:
    print("successfully detected!")

