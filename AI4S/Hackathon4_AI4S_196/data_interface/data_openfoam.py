#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time     : 2023/3/11
# @Author   : Hanwei Wang
# @Email    : wanghanweibnds2015@gmail.com
# @File     : data_openfoam.py

import os
from data_interface import DataInterface
import pandas as pd
import numpy as np
import fluidfoam as ff

__all__ = ['DataOpenfoam']


# OpenFOAM data
class DataOpenfoam(DataInterface):
    """
    Interface for inserting and processing data from OpenFOAM results
    """

    def __init__(self, related_path=None, dimension=2, time_dependent=False):
        super().__init__(dimension=dimension, time_dependent=time_dependent)
        self.related_path = related_path
        if self.related_path is None:
            current_path = os.getcwd()
            foam_file = pd.read_csv(current_path + "related_foam_path")
            self.related_path = foam_file['foam_path'][0]

    def read_mesh(self):
        """
        read mesh from OpenFOAM projects

        Examples:
        >>>foam_path = "~/OpenFOAM/$USER/run/cavity"
        >>>external_data = DataOpenfoam(foam_path)
        >>>external_data.read_mesh()
        :return: None
        """

        x, y, z = ff.readof.readmesh(self.related_path, verbose=False)
        xx = x.reshape(-1, 1)
        yy = y.reshape(-1, 1)
        zz = z.reshape(-1, 1)
        self.mesh = np.column_stack((xx, yy, zz))

    def read_field_data(self, *args):
        """
        read given field data from OpenFOAM projects

        Examples:
        >>>foam_path = "~/OpenFOAM/$USER/run/cavity"
        >>>external_data = DataOpenfoam(foam_path)
        >>>external_data.read_field_data('U','p')
        :param args: physical field to read, multiple fields to be give at the same time is allowed
        :return: None
        """

        if self.time_dependent is False:
            field = {}
            self.field_data[0] = field
            for arg in args:
                current_data = ff.readof.readfield(self.related_path, "latestTime", arg, verbose=False)
                field[arg] = current_data

        elif self.time_dependent is True:
            all_dir = os.listdir(self.related_path)
            all_dir.remove('0')
            for dir_name in all_dir:
                field = {}
                for arg in args:
                    file_name = self.related_path + '/' + dir_name + '/' + arg
                    if os.path.exists(file_name):
                        current_data = ff.readof.readfield(self.related_path, dir_name, arg, verbose=False)
                        field[arg] = current_data
                if field != {}:
                    self.field_data[float(dir_name)] = field

