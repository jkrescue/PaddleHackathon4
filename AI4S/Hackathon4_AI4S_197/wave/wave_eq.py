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
from pyDOE import lhs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from basic_model import DeepModelSingle
from parser_pinn import get_parser
from gen_wave_data import get_noise_data, get_truth
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


class PINN_wave(DeepModelSingle):
    def __init__(self, planes):
        super(PINN_wave, self).__init__(planes, active=nn.Tanh())
        self.c = paddle.create_parameter(shape=[1], dtype='float32', default_initializer=nn.initializer.Constant(1.0))
        self.add_parameter("c", self.c)

    def gradients(self, y, x):
        return paddle.grad(y, x, grad_outputs=paddle.ones_like(y),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    def get_c(self):
        return paddle.to_tensor(self.c, stop_gradient=False)

    def equation(self, inn_var):
        model = self
        u = model(inn_var)
        duda = self.gradients(u, inn_var)

        dudx, dudt = duda[..., 0:1], duda[..., 1:2]
        d2udx2 = self.gradients(dudx, inn_var)[..., 0:1]
        d2udt2 = self.gradients(dudt, inn_var)[..., 1:2]

        res_u = d2udt2 - self.c * d2udx2

        return res_u


    def train(self, X_u_train, X_f_train, u_train, Loss_data, Loss_PDE, weight, Optimizer, Scheduler, log_loss):

        model = self
        Optimizer.clear_grad()
        u_pred = model(X_u_train)
        res_u = self.equation(X_f_train)
        data_loss = Loss_data(u_pred, u_train)
        eqs_loss = Loss_PDE(res_u, paddle.zeros_like(res_u))

        loss_batch = data_loss + weight * eqs_loss
        loss_batch.backward()

        log_loss.append([data_loss.item(), eqs_loss.item()])

        Optimizer.step()
        Scheduler.step()

    def predict_error(self):
        model = self
        inn_var, u = get_truth()
        inn_var = paddle.to_tensor(inn_var, dtype='float32', stop_gradient=False)
        u_pred = model(inn_var)
        error_u = np.linalg.norm(u - u_pred.numpy(), 2) / np.linalg.norm(u, 2)
        lambda_c = self.c.numpy()
        error_c = (np.abs(lambda_c - 1.54 ** 2) / 1.54 ** 2)[0]
        return error_u, error_c, lambda_c.item()

    def plot_result(self, X_u_train, filename):
        model = self
        _shape = (401, 201)
        X_star, u_star = get_truth()
        X_star = paddle.to_tensor(X_star, dtype='float32', stop_gradient=False)
        u_pred = model(X_star)
        x = X_star.numpy()[:, 0].reshape(*_shape)
        t = X_star.numpy()[:, 1].reshape(*_shape)
        print(X_star.shape, '?')
        fig, ax = plt.subplots(3, 1, figsize=(10, 10), dpi=200)
        fig.set_tight_layout(True)
        ax[0].pcolormesh(t.ravel().reshape(*_shape), x.ravel().reshape(*_shape), u_pred.numpy().ravel().reshape(*_shape),
                         vmin=-1, vmax=1.)
        ax[1].scatter(X_u_train.numpy()[:, 1:2], X_u_train.numpy()[:, 0:1], zorder=100)
        ax[1].pcolormesh(t.ravel().reshape(*_shape), x.ravel().reshape(*_shape), u_star.ravel().reshape(*_shape),
                         vmin=-1, vmax=1.)
        ax[2].pcolormesh(t.ravel().reshape(*_shape), x.ravel().reshape(*_shape),
                         u_pred.numpy().ravel().reshape(*_shape) - u_star.ravel().reshape(*_shape), vmin=-0.1, vmax=0.1,
                         cmap='bwr')
        plt.savefig(filename)
        plt.close()

def plot_loss(x, y, label, title=None, color=None, marker=None):
        # sbn.set_style('ticks')
        # sbn.set(color_codes=True)
        font = {'weight': 'normal', 'size': 30}
        plt.plot(x, y, label=label, color=color, marker=marker)
        plt.semilogy()
        plt.grid(True)  # 添加网格
        plt.legend(loc="upper right", prop=font)
        plt.xlabel('iterations', font)
        plt.ylabel('loss value', font)
        plt.yticks(size=font["size"])
        plt.xticks(size=font["size"])
        plt.title(title, font)
        # plt.pause(0.001)

def run_experiment(epoch_num, noise_type, noise, loss_type, N, weight, _data=[], abnormal_size=0):
    if paddle.is_compiled_with_cuda():
        paddle.device.set_device('gpu:0')
    else:
        paddle.device.set_device('cpu')


    ## 模型设置
    planes = [2] + [num_neurons] * num_layers + [1]
    # Model
    Net_model = PINN_wave(planes=planes)
    # Loss
    if loss_type == "square":
        Loss_data = nn.MSELoss()
    elif loss_type == "l1":
        Loss_data = nn.L1Loss()
    else:
        raise NotImplementedError(f'Loss type {loss_type} not implemented.')
    Loss_PDE = nn.MSELoss()
    # 下降策略
    Scheduler = paddle.optimizer.lr.MultiStepDecay(0.001, [adam_iter*0.6, adam_iter*0.8], gamma=0.1)
    # 优化算法
    Optimizer = paddle.optimizer.Adam(learning_rate=Scheduler, parameters=Net_model.parameters(), beta1=0.8, beta2=0.9)
    # Optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=Net_model.parameters())


    ## 数据生成
    # Domain bounds
    lb = np.array([0, 0])
    ub = np.array([1., 2.]) * np.pi

    if len(_data) == 0:
        X_u_train, u_train = get_noise_data(N, noise_type, noise, size=abnormal_size)
        if Nf == 0:
            X_f_train = X_u_train
        else:
            X_f_train = lb + (ub - lb) * lhs(1, Nf)
            X_f_train = np.concatenate([X_u_train, X_f_train], axis=0)
        _data.append((X_u_train, X_f_train, u_train))
    X_u_train, X_f_train, u_train = _data[0]
    X_u_train = paddle.to_tensor(X_u_train, dtype='float32', stop_gradient=False)
    X_f_train = paddle.to_tensor(X_f_train, dtype='float32', stop_gradient=False)
    u_train = paddle.to_tensor(u_train, dtype='float32', stop_gradient=False)

    ## 执行训练过程
    log_loss = []
    print_freq = 20
    sta_time = time.time()

    for it in range(adam_iter):

        learning_rate = Optimizer.get_lr()
        Net_model.train(X_u_train, X_f_train, u_train, Loss_data, Loss_PDE, weight, Optimizer,Scheduler, log_loss)
        if (it + 1) % print_freq == 0:
            print('epoch: {:6d}, lr: {:.3e}, data_loss: {:.3e}, pde_loss: {:.3e}, c: {:.3f},cost: {:.2f}'.
              format(it, learning_rate, log_loss[-1][0], log_loss[-1][-1], Net_model.c.item(), time.time() - sta_time))
            plt.figure(100, figsize=(10, 6))
            plt.clf()
            plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'data_loss')
            plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'pde_loss', title='loss')
            plt.savefig(path.joinpath(f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_loss.png'))

    Scheduler._learning_rate = 1e-20
    Optimizer._learning_rate = 1e-20
    for it in range(100):
        Net_model.train(X_u_train, X_f_train, u_train, Loss_data, Loss_PDE, weight, Optimizer, Scheduler, log_loss)
    paddle.save({'epoch': adam_iter, 'log_loss': log_loss,
                 'model': Net_model.state_dict(), "optimizer": Optimizer.state_dict()},
        os.path.join(path, f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}.pdparams'))
    plt.figure(100, figsize=(10, 6))
    plt.clf()
    plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 0], 'data_loss')
    plot_loss(np.arange(len(log_loss)), np.array(log_loss)[:, 1], 'pde_loss', title='loss')
    plt.savefig(path.joinpath(f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_loss.png'))
    error_u, error_c, c = Net_model.predict_error()
    Net_model.plot_result(X_u_train, path.joinpath(f'{epoch_num}_{loss_type}_{N}_{noise_type}_{noise}_{abnormal_size}_{weight}_field.png'))
    with open(path.joinpath('result.csv'), 'a+') as f:
        f.write(f"{epoch_num},{loss_type},{N},{noise_type},{noise},{abnormal_size},{weight},{error_u},{error_c},{c}\n")

    logger.info(f"EC: {error_c * 100:.3f}%")

def get_last_idx(filename):
    if not os.path.exists(filename):
        return -1
    with open(filename, "r") as f1:
        last_idx = int(f1.readlines()[-1].strip().split(',')[0])
        return last_idx

# if __name__ == "__main__":
#     N = 1000
#     noise = 0
#     abnormal_ratio = 0.2
#     abnormal_size = int(N * abnormal_ratio)
#     noise_type = 'outlinear'
#     _data = []
#     weight = 1.0
#     run_experiment(0, N=N, noise=noise, noise_type=noise_type, weight=weight,
#                    loss_type='l1', _data=_data, abnormal_size=abnormal_size)