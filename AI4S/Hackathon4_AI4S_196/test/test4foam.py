#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time     : 2023/3/10
# @Author   : Hanwei Wang
# @Email    : wanghanweibnds2015@gmail.com
# @File     : test4foam.py

from data_openfoam import DataOpenfoam

path = "/home/dell/OpenFOAM/dell-v2206/run/cavity"
mydata = DataOpenfoam(related_path=path, dimension=2, time_dependent=True)
mydata.read_mesh()
# print(mydata.mesh)
mydata.read_field_data('U', 'p')
# print(mydata.field_data.keys())
# print(mydata.get_point_data(field='p', x=0.5, y=0.5, t=0.4, method="nearest"))
# print(mydata.get_line_data(field='U', x=[0.5, 0], y=[0.5, 1], t=0.4, method="nearest"))
print(mydata.field_data[0.5]['U'][0][0])
print(mydata.get_field('U', t=0.5, vector=True, direction='x'))
