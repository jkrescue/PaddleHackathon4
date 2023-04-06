#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time     : 2023/3/24
# @Author   : Hanwei Wang
# @Email    : wanghanweibnds2015@gmail.com
# @File     : read_cavity.py
import numpy as np

import data_openfoam as dtof

# OpenFOAM path
foam_path = "OpenFOAMCavity"

# read from OpenFoam file and insert as supervised data
of_data = dtof.DataOpenfoam(foam_path, dimension=2, time_dependent=True)

# get mesh
of_data.read_mesh()
print("mesh information: ")
print(of_data.mesh)
print("\n")

# read field
of_data.read_field_data("U", "p")

# get point data
point_data = of_data.get_point_data("U", x=0.06, y=0.03, t=0.4)
print("velocity U at (0.06,0.03) is:")
print(point_data)
print("\n")

# get line data
line_x = np.linspace(0, 0.1, 51)
line_y = np.linspace(0, 0.1, 51)
line_data = of_data.get_line_data("p", x=line_x, y=line_y, t=0.5)
print("pressure p along y=x is:")
print(line_data)
print("\n")
