#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time     : 2023/3/24
# @Author   : Hanwei Wang
# @Email    : wanghanweibnds2015@gmail.com
# @File     : insert_from_cavity.py

import paddle
import paddlescience as psci
import data_openfoam as dtof
import numpy as np
import os

paddle.seed(1)
np.random.seed(1)

# OpenFOAM path
foam_path = "OpenFOAMCavity"

# read from OpenFoam file and insert as supervised data
of_data = dtof.DataOpenfoam(foam_path, dimension=2)

# get point data

# internal data preparing
of_data.read_mesh()
internal_x = np.float32(of_data.mesh[:, 0])
internal_y = np.float32(of_data.mesh[:, 1])
internal_z = np.float32(of_data.mesh[:, 2])

# boundary top
boundary_top_x = np.linspace(0, 0.1, 21)
boundary_top_y = 0.1 * np.ones(21)
boundary_top_x = np.float32(boundary_top_x)
boundary_top_y = np.float32(boundary_top_y)

# boundary bottom
boundary_bottom_x = np.linspace(0, 0.1, 21)
boundary_bottom_y = np.zeros(21)
boundary_bottom_x = np.float32(boundary_bottom_x)
boundary_bottom_y = np.float32(boundary_bottom_y)

# boundary left
boundary_left_x = np.zeros(21)
boundary_left_y = np.linspace(0, 0.1, 21)
boundary_left_x = np.float32(boundary_left_x)
boundary_left_y = np.float32(boundary_left_y)

# boundary right
boundary_right_x = 0.1 * np.ones(21)
boundary_right_y = np.linspace(0.0, 0.1, 21)
boundary_right_x = np.float32(boundary_right_x)
boundary_right_y = np.float32(boundary_right_y)

# supervised data
of_data.read_field_data('U', 'p')
cfd_u = of_data.get_field("U", t=0, vector=True, direction='x')
cfd_v = of_data.get_field('U', t=0, vector=True, direction='y')
cfd_p = of_data.get_field('p', t=0, vector=False)
cfd_u = np.float32(cfd_u)
cfd_v = np.float32(cfd_v)
cfd_p = np.float32(cfd_p)


# data set
input_train = np.stack((internal_x, internal_y), axis=1)
input_bc_top = np.stack((boundary_top_x, boundary_top_y), axis=1)
input_bc_bottom = np.stack((boundary_bottom_x, boundary_bottom_y), axis=1)
input_bc_left = np.stack((boundary_left_x, boundary_left_y), axis=1)
input_bc_right = np.stack((boundary_right_x, boundary_right_y), axis=1)
input_cfd = np.stack((internal_x, internal_y), axis=1)
ref_cfd = np.stack((cfd_u, cfd_v, cfd_p), axis=1)

# N-S
pde = psci.pde.NavierStokes(nu=0.01, rho=1.0, dim=2, time_dependent=False, weight=0.0001)

# set boundary condition
weight_top_u = lambda x, y: 1.0 - 20.0 * abs(x-0.05)
bc_top_u = psci.bc.Dirichlet('u', rhs=1.0, weight=weight_top_u)
bc_top_v = psci.bc.Dirichlet('v', rhs=0)
bc_bottom_u = psci.bc.Dirichlet('u', rhs=0)
bc_bottom_v = psci.bc.Dirichlet('v', rhs=0)
bc_left_u = psci.bc.Dirichlet('u', rhs=0)
bc_left_v = psci.bc.Dirichlet('v', rhs=0)
bc_right_u = psci.bc.Dirichlet('u', rhs=0)
bc_right_v = psci.bc.Dirichlet('v', rhs=0)

# add boundary and boundary condition
pde.set_bc("top", bc_top_u, bc_top_v)
pde.set_bc("bottom", bc_bottom_u, bc_bottom_v)
pde.set_bc("left", bc_left_u, bc_left_v)
pde.set_bc("right", bc_right_u, bc_right_v)

# Network
net = psci.network.FCNet(num_ins=2, num_outs=3, num_layers=10, hidden_size=50, activation='tanh')

out_train = net(input_train)
out_bc_top = net(input_bc_top)
out_bc_bottom = net(input_bc_bottom)
out_bc_left = net(input_bc_left)
out_bc_right = net(input_bc_right)
out_cfd = net(input_cfd)

# equation loss
loss_eq1 = psci.loss.EqLoss(pde.equations[0], netout=out_train)
loss_eq2 = psci.loss.EqLoss(pde.equations[1], netout=out_train)
loss_eq3 = psci.loss.EqLoss(pde.equations[2], netout=out_train)
# bc loss
loss_bc_top = psci.loss.BcLoss("top", netout=out_bc_top)
loss_bc_bottom = psci.loss.BcLoss("bottom", netout=out_bc_bottom)
loss_bc_left = psci.loss.BcLoss("left", netout=out_bc_left)
loss_bc_right = psci.loss.BcLoss("right", netout=out_bc_right)
# supervise loss
loss_cfd = psci.loss.DataLoss(netout=out_cfd, ref=ref_cfd)

# total loss
loss = (loss_eq1 + loss_eq2 + loss_eq3) + (
        10 * loss_bc_top + loss_bc_bottom + 10 * loss_bc_left + 10 * loss_bc_right) + loss_cfd

# Algorithm
algo = psci.algorithm.PINNs(net=net, loss=loss)

# Optimizer
opt = psci.optimizer.Adam(learning_rate=0.001, parameters=net.parameters())

# Solver
solver = psci.solver.Solver(pde=pde, algo=algo, opt=opt)

# Solve
solution = solver.solve(num_epoch=10000)

# Save last time data to vtk
if not os.path.exists("results"):
    os.mkdir("results")
cord = input_train
psci.visu.__save_vtk_modified(filename="output_train", cordinate=cord, data=solution[0], title=['u', 'v', 'p'])
