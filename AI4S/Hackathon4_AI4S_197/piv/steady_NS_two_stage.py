import os
import sys
import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import paddle
import paddle.nn as nn
from paddle.incubate.optimizer import LBFGS
from pyDOE import lhs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from basic_model import DeepModelSingle
from parser_pinn import get_parser
from gen_piv_data import get_noise_data, get_truth, DelCylPT
import math
from scipy.special import erfinv
from logger import logger

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;

parser_PINN = get_parser()
args = parser_PINN.parse_args()
path = pathlib.Path(args.save_path)
path.mkdir(exist_ok=True, parents=True)
for key, val in vars(args).items():
    print(f"{key} = {val}")
with open(path.joinpath('config'), 'wt') as f:
    f.writelines([f"{key} = {val}\n" for key, val in vars(args).items()])
adam_iter: int = int(args.adam_iter)
bfgs_iter: int = int(args.bfgs_iter)
verbose: bool = bool(args.verbose)
repeat: int = int(args.repeat)
start_epoch: int = int(args.start_epoch)
Nf: int = int(args.Nf)
num_neurons: int = int(args.num_neurons)
num_layers: int = int(args.num_layers)


class PINN_laminar_flow(DeepModelSingle):
    def __init__(self, planes):
        super(PINN_laminar_flow, self).__init__(planes, active=nn.Tanh())
        self.rho = 1.0
        self.mu = 0.02

    def gradients(self, y, x):
        return paddle.grad(y, x, grad_outputs=paddle.ones_like(y),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    def equation(self, inn_var, out_var):
        p = out_var[..., 0:1]
        psi = out_var[..., 1:2]
        s11 = out_var[..., 2:3]
        s22 = out_var[..., 3:4]
        s12 = out_var[..., 4:5]
        u = self.gradients(psi, inn_var)[..., 1:2]
        v = -self.gradients(psi, inn_var)[..., 0:1]

        dpda = self.gradients(p, inn_var)
        duda = self.gradients(u, inn_var)
        dvda = self.gradients(v, inn_var)
        ds11da = self.gradients(s11, inn_var)
        ds22da = self.gradients(s22, inn_var)
        ds12da = self.gradients(s12, inn_var)

        dpdx, dpdy = dpda[..., 0:1], dpda[..., 1:2]
        dudx, dudy = dpda[..., 0:1], dpda[..., 1:2]
        dvdx, dvdy = dpda[..., 0:1], dpda[..., 1:2]
        ds11dx, ds11dy = ds11da[..., 0:1], ds11da[..., 1:2]
        ds22dx, ds22dy = ds22da[..., 0:1], ds22da[..., 1:2]
        ds12dx, ds12dy = ds12da[..., 0:1], ds12da[..., 1:2]

        res_u = self.rho * (u * dudx + v * dudy) - ds11dx - ds12dy
        res_v = self.rho * (u * dvdx + v * dvdy) - ds12dx - ds22dy
        # res_c = dudx + dvdy
        res_s11 = -p + 2 * self.mu * dudx - s11
        res_s22 = -p + 2 * self.mu * dvdy - s22
        res_s12 = self.mu * (dudy + dvdx) - s12
        res_p = p + (s11 + s22) / 2

        return paddle.concat((res_u, res_v, res_s11, res_s22, res_s12, res_p), axis=-1), \
               paddle.concat((p, u, v), axis=-1)  # cat给定维度，stack新维度

    def inference(self, inn_var):

        model = self
        out_pred = model(inn_var)
        equation, field_pred = model.equation(inn_var, out_pred)

        return equation, field_pred

    def train(self, XY_c, OUTLET, WALL, DATA, Loss_PDE, weight, switch, Optimizer, log_loss):

        model = self
        Optimizer.clear_grad()
        inn_data = DATA[:, 0:2]
        out_data = DATA[:, 2:4]
        _, field_data_pred = self.inference(inn_data)
        res_i, field_c_pred = self.inference(XY_c)
        _, field_outlet_pred = self.inference(OUTLET)
        _, field_wall_pred = self.inference(WALL)
        uv_data_pre = field_data_pred[..., 1:3]
        p_outlet_pre = field_outlet_pred[..., 0:1]
        uv_wall_pre = field_wall_pred[..., 1:3]
        data_loss = (1 - switch) * nn.L1Loss()(out_data, uv_data_pre) + switch * nn.MSELoss()(out_data, uv_data_pre)
        eqs_loss = Loss_PDE(res_i, paddle.zeros_like(res_i))
        wall_loss = Loss_PDE(uv_wall_pre, paddle.zeros_like(uv_wall_pre))
        outlet_loss = Loss_PDE(p_outlet_pre, paddle.zeros_like(p_outlet_pre))

        loss_batch = data_loss + weight * eqs_loss + wall_loss + outlet_loss
        loss_batch.backward()

        log_loss.append([data_loss.item(), eqs_loss.item(), wall_loss.item(), outlet_loss.item()])

        Optimizer.step()

    def train_bfgs(self, XY_c, OUTLET, WALL, DATA, Loss_PDE, weight, switch, Optimizer, log_loss):
        def closure():
            model = self
            Optimizer.clear_grad()
            inn_data = DATA[:, 0:2]
            out_data = DATA[:, 2:4]
            _, field_data_pred = self.inference(inn_data)
            res_i, field_c_pred = self.inference(XY_c)
            _, field_outlet_pred = self.inference(OUTLET)
            _, field_wall_pred = self.inference(WALL)
            uv_data_pre = field_data_pred[..., 1:3]
            p_outlet_pre = field_outlet_pred[..., 0:1]
            uv_wall_pre = field_wall_pred[..., 1:3]
            data_loss = (1 - switch) * nn.L1Loss()(out_data, uv_data_pre) + switch * nn.MSELoss()(out_data, uv_data_pre)
            eqs_loss = Loss_PDE(res_i, paddle.zeros_like(res_i))
            wall_loss = Loss_PDE(uv_wall_pre, paddle.zeros_like(uv_wall_pre))
            outlet_loss = Loss_PDE(p_outlet_pre, paddle.zeros_like(p_outlet_pre))

            loss_batch = data_loss + weight * eqs_loss + wall_loss + outlet_loss
            loss_batch.backward()

            log_loss.append([data_loss.item(), eqs_loss.item(), wall_loss.item(), outlet_loss.item()])
            return loss_batch

        Optimizer.step(closure)

    def sieve_obs(self, DATA, ratio):
        model = self
        inn_data = DATA[:, 0:2]
        u_data = DATA[:, 2:3]
        v_data = DATA[:, 3:4]
        _, field_data_pred = self.inference(inn_data)
        u_data_pre = field_data_pred[..., 1:2]
        v_data_pre = field_data_pred[..., 2:3]
        data_erro = paddle.sqrt(paddle.square(u_data_pre - u_data) + paddle.square(v_data_pre - v_data)).numpy()
        error = data_erro.ravel()
        # print("data erro:", error)
        execluded_number = int(len(error) * ratio)
        print(f'{execluded_number} observations have been removed.')
        ind_valid = np.argpartition(error, -execluded_number)[:-execluded_number]

        x_data = DATA.numpy()[ind_valid, 0:1]
        # print("sieve:", x_data.shape)
        return ind_valid

    def sieve_obs_sigma(self, DATA, k=2):
        model = self
        inn_data = DATA[:, 0:2]
        u_data = DATA[:, 2:3]
        v_data = DATA[:, 3:4]
        _, field_data_pred = self.inference(inn_data)
        u_data_pre = field_data_pred[..., 1:2]
        v_data_pre = field_data_pred[..., 2:3]
        data_erro = paddle.sqrt(paddle.square(u_data_pre - u_data) + paddle.square(v_data_pre - v_data)).numpy()
        error = data_erro.ravel()
        sigma = np.median(np.abs(error)) / math.sqrt(2) / erfinv(0.5)
        ind_valid = (np.abs(error) <= k * sigma)
        # print("data erro:", error)
        original_N = len(error)

        x_data = DATA.numpy()[ind_valid, 0:1]
        new_N = len(x_data)
        print(f'{original_N - new_N} observations have been removed.')
        # print("sieve:", x_data.shape)
        return ind_valid

    def loadmodel(self, File):
        """
        神经网络模型权重读入
        """
        try:
            checkpoint = paddle.load(File)
            self.set_state_dict(checkpoint['model'])  # 从字典中依次读取
            start_epoch = checkpoint['epoch']
            print("load start epoch at epoch " + str(start_epoch))
        except:
            print("load model failed！ start a new model.")

        return

    def predict_error(self):
        model = self
        x, y, u, v, p = get_truth(pressure=True)
        inn_var = paddle.to_tensor(np.concatenate((x, y), axis=1), dtype='float32', stop_gradient=False)
        _, field_pred = self.inference(inn_var)
        out_pred = field_pred.numpy()
        error_u = np.linalg.norm(out_pred[:, (1,)] - u, 2) / np.linalg.norm(u, 2)
        error_v = np.linalg.norm(out_pred[:, (2,)] - v, 2) / np.linalg.norm(v, 2)
        error_vel = np.sqrt(np.sum((out_pred[:, (1,)] - u) ** 2 + (out_pred[:, (2,)] - v) ** 2)) / np.sqrt(
            np.sum(u ** 2 + v ** 2))
        error_max = np.max(np.sqrt((out_pred[:, (1,)] - u) ** 2 + (out_pred[:, (2,)] - v) ** 2))
        error_p = np.linalg.norm(out_pred[:, (0,)] - p, 2) / np.linalg.norm(p, 2)
        return error_u, error_v, error_vel, error_max, error_p

    def plot_result(self, DATA, filename):
        model = self
        x_train, y_train, u_train, v_train, p_train = get_truth(pressure=True)
        inn_var = paddle.to_tensor(np.concatenate((x_train, y_train), axis=1), dtype='float32', stop_gradient=False)
        _, field_pred = self.inference(inn_var)
        out_pred = field_pred.numpy()
        plt.clf()
        fig, ax = plt.subplots(7, 1, figsize=(10, 23))
        fig.set_tight_layout(True)
        im = ax[0].scatter(x_train, y_train, c=np.hypot(u_train, v_train))
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[1].scatter(x_train, y_train, c=np.hypot(out_pred[:, (1,)], out_pred[:, (2,)]))
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[2].scatter(x_train, y_train, c=np.hypot(u_train - out_pred[:, (1,)], v_train - out_pred[:, (2,)]),
                           vmin=0.,
                           vmax=0.5)
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax[2].set_title(f'mse: {np.mean(np.hypot(u_train - out_pred[:, (1,)], v_train - out_pred[:, (2,)]) ** 2):.4f}')

        im = ax[3].scatter(x_train, y_train, c=p_train)
        divider = make_axes_locatable(ax[3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[4].scatter(x_train, y_train, c=out_pred[:, (0,)], vmin=0, vmax=3.6)
        divider = make_axes_locatable(ax[4])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        im = ax[5].scatter(x_train, y_train, c=np.abs(p_train - out_pred[:, (0,)]), vmin=0, vmax=3.6)
        divider = make_axes_locatable(ax[5])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax[5].set_title(f'mse: {np.mean(np.abs(out_pred[:, (0,)] - p_train)):.4f}')
        plt.colorbar(im, cax=cax)

        im = ax[6].scatter(DATA[:, 0:1].numpy().ravel(), DATA[:, 1:2].numpy().ravel(), s=4)

        plt.savefig(filename)


def run_experiment(epoch_num, noise_type, noise, loss_type, N, weight, _data=[], abnormal_size=0, sieve_sigma=0,
                   sieve_ratio=0):
    if paddle.is_compiled_with_cuda():
        paddle.device.set_device('gpu:0')
    else:
        paddle.device.set_device('cpu')

    print(f"运行算例：{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_{sieve_sigma}_{sieve_ratio}")
    ## 模型设置
    planes = [2] + [num_neurons] * num_layers + [5]
    # Model
    Net_model = PINN_laminar_flow(planes=planes)

    Loss_PDE = nn.MSELoss()
    # 下降策略
    # Scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.001, milestones=[1000, ], gamma=0.1)
    # 优化算法
    # Optimizer = paddle.optimizer.Adam(learning_rate=Scheduler, parameters=Net_model.parameters(), beta1=0.8, beta2=0.9)
    Optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=Net_model.parameters())

    ## 数据生成
    # Domain bounds
    lb = np.array([0, 0])
    ub = np.array([1.1, 0.41])

    # Network configuration
    uv_layers = [2] + 8 * [40] + [5]

    wall_up = [0.0, 0.41] + [1.1, 0.0] * lhs(2, 101)
    wall_lw = [0.0, 0.00] + [1.1, 0.0] * lhs(2, 101)
    OUTLET = [1.1, 0.0] + [0.0, 0.41] * lhs(2, 201)

    # Cylinder surface
    r = 0.05
    theta = [0.0] + [2 * np.pi] * lhs(1, 360)
    x_CYLD = np.multiply(r, np.cos(theta)) + 0.2
    y_CYLD = np.multiply(r, np.sin(theta)) + 0.2
    CYLD = np.concatenate((x_CYLD, y_CYLD), 1)
    WALL = np.concatenate((CYLD, wall_up, wall_lw), 0)

    # Collocation point for equation residual
    XY_c = lb + (ub - lb) * lhs(2, 40000)
    XY_c_refine = [0.1, 0.1] + [0.2, 0.2] * lhs(2, 10000)
    XY_c = np.concatenate((XY_c, XY_c_refine), 0)
    XY_c = DelCylPT(XY_c, xc=0.2, yc=0.2, r=0.05)

    # Visualize the collocation points
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # plt.scatter(XY_c[:,0:1], XY_c[:,1:2], marker='o', alpha=0.1 ,color='blue')
    plt.scatter(WALL[:, 0:1], WALL[:, 1:2], marker='o', alpha=0.2, color='green')
    plt.scatter(OUTLET[:, 0:1], OUTLET[:, 1:2], marker='o', alpha=0.2, color='orange')

    if len(_data) == 0:
        x_train, y_train, u_train, v_train = get_noise_data(N=N, noise_type=noise_type, sigma=noise, size=abnormal_size)
        _data.append((x_train, y_train, u_train, v_train))
    DATA = np.concatenate(_data[0], axis=1)
    DATA = paddle.to_tensor(DATA, dtype='float32', stop_gradient=False)
    WALL = paddle.to_tensor(WALL, dtype='float32', stop_gradient=False)
    OUTLET = paddle.to_tensor(OUTLET, dtype='float32', stop_gradient=False)
    XY_c = paddle.to_tensor(XY_c, dtype='float32', stop_gradient=False)
    plt.scatter(_data[0][0].flatten(), _data[0][1].flatten(), marker='o', alpha=0.2, color='blue')
    plt.savefig('collocation.png')

    ## 执行训练过程
    log_loss = []
    print_freq = 20
    save_freq = 20
    sta_time = time.time()

    if (sieve_sigma == 0) and (sieve_ratio == 0):
        if loss_type == 'l1':
            switch = 0
        else:
            switch = 1
        # Adam初步优化
        for it in range(adam_iter):
            learning_rate = Optimizer.get_lr()
            Net_model.train(XY_c, OUTLET, WALL, DATA, Loss_PDE, weight, switch, Optimizer, log_loss)
            if (it + 1) % print_freq == 0:
                print('epoch: {:6d}, lr: {:.3e}, data_loss: {:.3e}, pde_loss: {:.3e}, wall_loss: {:.3e}, outlet_loss:'
                      ' {:.3e}, cost: {:.2f}'.format(it, learning_rate, log_loss[-1][0], log_loss[-1][1],
                                                     log_loss[-1][2], log_loss[-1][3], time.time() - sta_time))
                paddle.save({'epoch': adam_iter, 'log_loss': log_loss,
                             'model': Net_model.state_dict(), "optimizer": Optimizer.state_dict()},
                            os.path.join(path,
                                f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}.pdparams'))
        # LBFGS优化
        Optimizer = LBFGS(parameters=Net_model.parameters())
        for it in range(bfgs_iter):
            Net_model.train_bfgs(XY_c, OUTLET, WALL, DATA, Loss_PDE, weight, switch, Optimizer, log_loss)
            if (it + 1) % print_freq == 0:
                print('epoch: {:6d}, data_loss: {:.3e}, pde_loss: {:.3e}, cost: {:.2f}'.
                      format(it, log_loss[-1][0], log_loss[-1][-1], time.time() - sta_time))
                error_u = Net_model.predict_error()
                print(error_u)
                paddle.save({'epoch': adam_iter, 'log_loss': log_loss,
                             'model': Net_model.state_dict(), "optimizer": Optimizer.state_dict()},
                            os.path.join(path,
                                         f'{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_0_0.pdparams'))
        # Adam终优化
        Optimizer = paddle.optimizer.Adam(learning_rate=1e-20, parameters=Net_model.parameters())
        for it in range(100):
            Net_model.train(XY_c, OUTLET, WALL, DATA, Loss_PDE, weight, switch, Optimizer, log_loss)
    else:
        Net_model.loadmodel(str(path) + '/' + f"l1_{N}_{noise_type}_{noise}_{abnormal_size}_1.0_0_0.pdparams")
        if sieve_sigma > 0 and sieve_ratio == 0:
            ind_valid = Net_model.sieve_obs_sigma(DATA, k=sieve_sigma)
            DATA = DATA.numpy()[ind_valid, :]
            DATA = paddle.to_tensor(DATA, dtype='float32', stop_gradient=False)
        elif sieve_sigma == 0 and sieve_ratio > 0:
            ind_valid = Net_model.sieve_obs(DATA, ratio=sieve_ratio)
            DATA = DATA.numpy()[ind_valid, :]
            DATA = paddle.to_tensor(DATA, dtype='float32', stop_gradient=False)
        else:
            print(sieve_sigma, sieve_ratio)
            raise Exception('not good')
        assert loss_type == 'square'
        switch = 1
        # Adam初步优化
        for it in range(1000):
            learning_rate = Optimizer.get_lr()
            Net_model.train(XY_c, OUTLET, WALL, DATA, Loss_PDE, weight, switch, Optimizer, log_loss)
            if (it + 1) % print_freq == 0:
                print('epoch: {:6d}, lr: {:.3e}, data_loss: {:.3e}, pde_loss: {:.3e}, wall_loss: {:.3e}, outlet_loss:'
                      ' {:.3e}, cost: {:.2f}'.format(it, learning_rate, log_loss[-1][0], log_loss[-1][1],
                                                     log_loss[-1][2], log_loss[-1][3], time.time() - sta_time))
                paddle.save({'epoch': adam_iter, 'log_loss': log_loss,
                             'model': Net_model.state_dict(), "optimizer": Optimizer.state_dict()},
                            os.path.join(path, f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_'
                                               f'{weight}_{sieve_sigma}_{sieve_ratio}.pdparams'))
        # LBFGS优化
        Optimizer = LBFGS(parameters=Net_model.parameters())
        for it in range(bfgs_iter):
            Net_model.train_bfgs(XY_c, OUTLET, WALL, DATA, Loss_PDE, weight, switch, Optimizer, log_loss)
            if (it + 1) % print_freq == 0:
                print('epoch: {:6d}, data_loss: {:.3e}, pde_loss: {:.3e}, cost: {:.2f}'.
                      format(it, log_loss[-1][0], log_loss[-1][-1], time.time() - sta_time))
                error_u = Net_model.predict_error()
                print(error_u)
                paddle.save({'epoch': adam_iter, 'log_loss': log_loss,
                             'model': Net_model.state_dict(), "optimizer": Optimizer.state_dict()},
                            os.path.join(path,
                                         f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}.pdparams'))
        Optimizer = paddle.optimizer.Adam(learning_rate=1e-20, parameters=Net_model.parameters())
        # Adam终优化
        for it in range(100):
            Net_model.train(XY_c, OUTLET, WALL, DATA, Loss_PDE, weight, switch, Optimizer, log_loss)
    paddle.save({'epoch': adam_iter, 'log_loss': log_loss,
                 'model': Net_model.state_dict(), "optimizer": Optimizer.state_dict()},
                os.path.join(path, f'{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_{sieve_sigma}'
                                   f'_{sieve_ratio}.pdparams'))
    error = Net_model.predict_error()
    Net_model.plot_result(DATA,
                          path.joinpath(
                              f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_{sieve_sigma}'
                              f'_{sieve_ratio}.png'))
    with open(path.joinpath('result.csv'), 'a+') as f:
        f.write(f"{epoch_num},{loss_type},{N},{noise_type},{noise},{abnormal_size},{weight},{error},{sieve_sigma},"
                f"{sieve_ratio},{len(DATA[:, 0:1])}\n")

    print("--- %s seconds ---" % (time.time() - sta_time))


def get_last_idx(filename):
    if not os.path.exists(filename):
        return -1
    with open(filename, "r") as f1:
        last_idx = int(f1.readlines()[-1].strip().split(',')[0])
        return last_idx


if __name__ == "__main__":
    idx = 0
    last_idx = get_last_idx(path.joinpath('result.csv'))
    idx = last_idx + 1
    executed_flag = False

    for r in range(repeat):
        for noise_type in ['outlinear']:
            for noise, abnormal_ratio in zip([0.20], [0.20]):
                for N in [500]:
                    for loss_type, sieve_sigma, sieve_ratio in zip(
                            ['l1', 'square', 'square', 'square', 'square', 'square', 'square', 'square'],
                            [0, 0, 0, 0, 0, 2, 2.5, 3],
                            [0, 0, 0.1, 0.2, 0.3, 0, 0, 0]):
                        _data = []
                        abnormal_size = int(N * abnormal_ratio)
                        if executed_flag:
                            sys.exit()
                        for weight in [1E-0]:
                            if idx > last_idx:
                                run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=weight,
                                               loss_type=loss_type, _data=_data, abnormal_size=abnormal_size,
                                               sieve_sigma=sieve_sigma, sieve_ratio=sieve_ratio)
                                # executed_flag = True
                            idx += 1
    if executed_flag:
        sys.exit()
    for r in range(repeat):
        for noise_type in ['contamined', 'normal', 't1', 'none']:
            for noise in [0.20]:
                for abnormal_ratio in [0]:
                    for N in [500]:
                        _data = []
                        for loss_type, sieve_sigma, sieve_ratio in zip(
                                ['l1', 'square', 'square', 'square', 'square', 'square', 'square', 'square', ],
                                [0, 0, 0, 0, 0, 2, 2.5, 3, ],
                                [0, 0, 0.1, 0.2, 0.3, 0, 0, 0, ]):
                            abnormal_size = int(N * abnormal_ratio)
                            if executed_flag:
                                sys.exit()
                            for weight in [1E-0]:
                                if idx > last_idx:
                                    run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=weight,
                                                   loss_type=loss_type, _data=_data, abnormal_size=abnormal_size,
                                                   sieve_sigma=sieve_sigma, sieve_ratio=sieve_ratio)
                                    # executed_flag = True
                                idx += 1
    if executed_flag:
        sys.exit()
    for r in range(repeat):
        for noise_type in ['outlinear']:
            for noise in [0.0]:
                for abnormal_ratio in [0.20]:
                    for N in [500]:
                        for loss_type, sieve_sigma, sieve_ratio in zip(
                                ['l1', 'square', 'square', 'square', 'square', 'square', 'square', 'square'],
                                [0, 0, 0, 0, 0, 2, 2.5, 3],
                                [0, 0, 0.1, 0.2, 0.3, 0, 0, 0]):
                            _data = []
                            abnormal_size = int(N * abnormal_ratio)
                            if executed_flag:
                                sys.exit()
                            for weight in [1E-0]:
                                if idx > last_idx:
                                    run_experiment(idx, N=N, noise=noise, noise_type=noise_type, weight=weight,
                                                   loss_type=loss_type, _data=_data, abnormal_size=abnormal_size,
                                                   sieve_sigma=sieve_sigma, sieve_ratio=sieve_ratio)
                                    # executed_flag = True
                                idx += 1
    if executed_flag:
        sys.exit()

    time.sleep(5)
