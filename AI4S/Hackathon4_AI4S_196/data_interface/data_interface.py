#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time     : 2023/3/11
# @Author   : Hanwei Wang
# @Email    : wanghanweibnds2015@gmail.com
# @File     : data_openfoam.py

from mesh import Mesh
from scipy.interpolate import griddata

__all__ = ["DataInterface"]


class DataInterface:
    """
    Interface for inserting and processing data from external platforms
    """

    # initialization
    def __init__(self, dimension=2, time_dependent=False):
        # dimension
        self.dimension = dimension
        # time dependency
        self.time_dependent = time_dependent
        # mesh
        self.mesh = Mesh(dimension, dynamic=None)
        # field data
        self.field_data = {}

    def get_point_data(self, field, x, y, z=0, t=0, method="nearest"):
        """
        return interpolated field data

        Examples:
        >>> external_data = DataInterface()
        >>> external_data.get_point_data(field='U', x=0.5, y=0.5, t=0, method="nearest")
        :param field: physics field to be obtained
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate if 3D case
        :param t: time
        :param method: interpolated method ("nearest", "linear", "cubic")
        :return: interpolated field data
        """

        this_field = self.field_data[t][field]
        count = this_field.shape[0]
        if len(this_field.shape) == 1:
            if self.dimension == 2:
                point_data = griddata(self.mesh[:, :2], this_field, (x, y), method=method)
            elif self.dimension == 3:
                point_data = griddata(self.mesh, this_field, (x, y, z), method=method)
        else:
            point_data = []
            if self.dimension == 2:
                for i in range(count):
                    current_point_data = griddata(self.mesh[:, :2], this_field[i], (x, y), method=method)
                    point_data.append(current_point_data)
            elif self.dimension == 3:
                for i in range(count):
                    current_point_data = griddata(self.mesh, this_field[i], (x, y, z), method=method)
                    point_data.append(current_point_data)
        return point_data

    def get_line_data(self, field, x, y, z=0, t=0, method="nearest"):
        """
        return interpolated field data

        Examples:
        >>> external_data = DataInterface()
        >>> external_data.get_line_data(field='U', x=[0.5, 1, 2], y=[0.5, 0, -1], t=0, method="nearest")
        :param field: physics field to be obtained
        :param x: x coordinate series
        :param y: y coordinate series
        :param z: z coordinate if 3D case
        :param t: time
        :param method: interpolated method ("nearest", "linear", "cubic")
        :return: interpolated field data
        """

        return self.get_point_data(field=field, x=x, y=y, z=z, t=t, method=method)

    def get_surface_data(self, field, x, y, z, t=0, method="nearest"):
        """
        return interpolated field data

        Examples:
        >>> external_data = DataInterface()
        >>> external_data.get_surface_data(field='U', x=[0, 1, 1, 0], y=[0, 0, 1, 1], z=[0, 0, 0, 0], t=0, method="nearest")
        :param field: physics field to be obtained
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        :param t: time
        :param method: interpolated method ("nearest", "linear")
        :return: interpolated field data
        """

        return self.get_point_data(field=field, x=x, y=y, z=z, t=t, method=method)

    def get_field(self, field, t=0.0, vector=False, direction='x'):
        """
        return specified field

        Examples:
        >>> external_data = DataInterface()
        >>> external_data.get_field('U', t=0.4, vector=True, direction='y')
        :param field: physics field to be obtained
        :param t: time, 0 if steady case
        :param vector: vector (True) or scalar (False)
        :param direction: direction ('x','y' or 'z') if vector
        :return: specified field
        """

        if vector:
            if direction == 'x':
                return self.field_data[t][field][0]
            if direction == 'y':
                return self.field_data[t][field][1]
            if direction == 'z':
                return self.field_data[t][field][2]
        else:
            return self.field_data[t][field]
