#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time     : 2023/3/10
# @Author   : Hanwei Wang
# @Email    : wanghanweibnds2015@gmail.com
# @File     : mesh.py

import numpy
import numpy as np

__all__ = ["Mesh"]


class Mesh:
    """
    store mesh information and operate
    """

    def __init__(self, dimension=2, dynamic=None):
        # dimension
        self.dimension = dimension
        # static/dynamic
        self.dynamic = dynamic
        # points information
        self.grid = None

    # dimension check
    @staticmethod
    def is_2d_grid(*args):
        """
        determine if args is a 2d-grid

        Examples:
            >>> M = Mesh()
            >>> M.is_2d_grid(np.mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]))
        :param args:  grid matrix: numpy.array, numpy.matrix, list are supported
        :return: True if given grid is 2D, False if given grid is 3D
        """
        grid = args[0]
        if isinstance(grid, (list, numpy.ndarray, numpy.matrix)):
            if grid.shape[1] == 2:
                return True
            elif grid.shape[1] == 3:
                return False
            else:
                raise IndexError
        else:
            raise TypeError

    def add_grid(self, *args):
        """
        add grid information when grid already exists

        Examples:
            >>> M = Mesh()
            >>> grid_origin = np.array([[1, 2], [2, 2]])
            >>> grid2add = np.array([[0, 0], [6, 6]])
            >>> M.set_grid(grid_origin)
            >>> M.add_grid(grid2add)
        :param args: grid matrix: numpy.array, numpy.matrix, list are supported
        :return: None
        """
        new_grid = args[0]
        if self.dimension == 2:
            if self.is_2d_grid(new_grid):
                self.grid = np.vstack([self.grid, new_grid])
            else:
                raise IndexError
        elif self.dimension == 3:
            if not self.is_2d_grid(new_grid):
                self.grid = np.vstack([self.grid, new_grid])
            else:
                raise IndexError

    def set_grid(self, *args):
        """
        add grid information when no grid exists, overwrite existed grid

        Examples:
            >>> M = Mesh()
            >>> grid_origin = np.array([[1, 2], [2, 2]])
            >>> M.set_grid(grid_origin)
        :param args: grid matrix: numpy.array, numpy.matrix, list are supported
        :return: None
        """
        new_grid = args[0]
        if self.dimension == 2:
            if self.is_2d_grid(new_grid):
                self.grid = new_grid
            else:
                raise IndexError
        elif self.dimension == 3:
            if not self.is_2d_grid(new_grid):
                self.grid = new_grid
            else:
                raise IndexError
