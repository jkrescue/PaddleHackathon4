import os
import sys
import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import math
import pathlib
import paddle
import paddle.nn as nn
import paddle.static as static
from pyDOE import lhs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from basic_model import DeepModelSingle, DeepModelMulti
from parser_pinn import get_parser
from gen_ns_data import get_noise_data, get_truth
from scipy.special import erfinv

def plot_result(p_pre1, p_pre2, p_pre3, filename):
    x, y, t, u, v, p = get_truth()
    _shape = (50, 100)

    plt.clf()
    fig, ax = plt.subplots(4, 3, figsize=(30, 20))
    fig.set_tight_layout(True)
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30}
    for i in range(3):
        t_p = (i + 1) * 5

        x_i = x[t_p::200, 0]
        y_i = y[t_p::200, 0]
        p_i = p[t_p::200, 0]
        p_pre_slice_1 = p_pre1[t_p::200, 0]
        p_pre_slice_2 = p_pre2[t_p::200, 0]
        p_pre_slice_3 = p_pre3[t_p::200, 0]
        p_mean_slice1 = np.mean(p_pre_slice_1 - p_i)
        p_mean_slice2 = np.mean(p_pre_slice_2 - p_i)
        p_mean_slice3 = np.mean(p_pre_slice_3 - p_i)

        im = ax[0][i].pcolormesh(x_i.ravel().reshape(*_shape), y_i.ravel().reshape(*_shape), p_i.ravel().reshape(*_shape),
                              shading='gouraud')
        ax[0][i].set_title(f't={t_p:.1f}', font)
        if i == 2:
            divider = make_axes_locatable(ax[0][i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=15)

        im = ax[1][i].pcolormesh(x_i.ravel().reshape(*_shape), y_i.ravel().reshape(*_shape),
                              (p_pre_slice_1 - p_mean_slice1).ravel().reshape(*_shape) - p_i.ravel().reshape(*_shape), cmap='bwr',
                              vmin=-0.08, vmax=0.08, shading='gouraud')
        if i == 2:
            divider = make_axes_locatable(ax[1][i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=15)

        im = ax[2][i].pcolormesh(x_i.ravel().reshape(*_shape), y_i.ravel().reshape(*_shape),
                                 (p_pre_slice_2 - p_mean_slice2).ravel().reshape(*_shape) - p_i.ravel().reshape(*_shape),
                                 vmin=-0.08, vmax=0.08, cmap='bwr',
                                 shading='gouraud')
        if i == 2:
            divider = make_axes_locatable(ax[2][i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=15)

        im = ax[3][i].pcolormesh(x_i.ravel().reshape(*_shape), y_i.ravel().reshape(*_shape),
                                 (p_pre_slice_3 - p_mean_slice3).ravel().reshape(*_shape) - p_i.ravel().reshape(*_shape),
                                 cmap='bwr',vmin=-0.08, vmax=0.08,
                                 shading='gouraud')
        if i == 2:
            divider = make_axes_locatable(ax[3][i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=15)
        ax[0][i].set_xticks([2,4,6,8], fontproperties='Times New Roman', size=15)
        ax[0][i].set_yticks([-2,0,2], fontproperties='Times New Roman', size=15)
        ax[1][i].set_xticks([2,4,6,8], fontproperties='Times New Roman', size=15)
        ax[1][i].set_yticks([-2,0,2], fontproperties='Times New Roman', size=15)
        ax[2][i].set_xticks([2,4,6,8], fontproperties='Times New Roman', size=15)
        ax[2][i].set_yticks([-2,0,2], fontproperties='Times New Roman', size=15)
        ax[3][i].set_xticks([2,4,6,8], fontproperties='Times New Roman', size=15)
        ax[3][i].set_yticks([-2,0,2], fontproperties='Times New Roman', size=15)
    ax[0][0].set_ylabel(f'exact', font)
    ax[1][0].set_ylabel(f'OLS-PINN(erro)', font)
    ax[2][0].set_ylabel(f'LAD-PINN(erro)', font)
    ax[3][0].set_ylabel(f'MAD-PINN2.0(erro)', font)
    plt.savefig(filename)

if __name__ == "__main__":
    L1_pred = np.load('data/L1_pred.pdparams', allow_pickle=True)
    L2_pred = np.load('data/L2_pred.pdparams', allow_pickle=True)
    MAD_pred = np.load('data/MAD_pred.pdparams', allow_pickle=True)
    plot_result(L2_pred[0], L1_pred[0], MAD_pred[0], 'p_erroVS_none.png')
